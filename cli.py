#!/usr/bin/env python3
"""
Command Line Interface for Docling RAG Agent.

Enhanced CLI with colors, formatting, and improved user experience.
"""

import asyncio
import asyncpg
import argparse
import logging
import os
import re
import sys
from typing import List, Dict, Any
from datetime import datetime

from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext

# Load environment variables
load_dotenv(".env")

logger = logging.getLogger(__name__)

# ANSI color codes for better formatting
class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


# Global database pool
db_pool = None
STOP_WORDS = {
    "the", "and", "for", "with", "from", "that", "this", "what", "when", "where", "which",
    "should", "would", "could", "about", "into", "during", "phase", "list", "only", "your",
    "have", "has", "are", "was", "were", "why", "how", "who", "whom", "their", "them",
    "then", "than", "also", "any", "all", "can", "may", "our", "you", "use", "using",
}


def resolve_agent_model(raw_model: str) -> str:
    """
    Normalize model string for PydanticAI.

    For local OpenAI-compatible endpoints (e.g., Ollama), model names often contain
    colons (e.g., phi4-mini:latest). Prefix them with openai: unless a known provider
    prefix is already present.
    """
    model = (raw_model or "").strip()
    if not model:
        return "openai:gpt-4o-mini"

    known_prefixes = (
        "openai:",
        "anthropic:",
        "google:",
        "groq:",
        "mistral:",
        "bedrock:",
        "azure:",
        "vertex:",
    )
    if model.startswith(known_prefixes):
        return model

    if os.getenv("OPENAI_BASE_URL"):
        return f"openai:{model}"

    return model


def extract_keywords(query: str, max_keywords: int = 10) -> List[str]:
    """Extract simple lexical keywords for hybrid retrieval."""
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", (query or "").lower())
    keywords: List[str] = []
    seen = set()
    for token in tokens:
        if token in STOP_WORDS:
            continue
        if token in seen:
            continue
        seen.add(token)
        keywords.append(token)
        if len(keywords) >= max_keywords:
            break
    return keywords


def normalize_for_dedupe(text: str) -> str:
    """Normalize chunk text for near-duplicate suppression."""
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


async def initialize_db():
    """Initialize database connection pool."""
    global db_pool
    if not db_pool:
        db_pool = await asyncpg.create_pool(
            os.getenv("DATABASE_URL"),
            min_size=2,
            max_size=10,
            command_timeout=60
        )
        # logger.info("Database connection pool initialized")


async def close_db():
    """Close database connection pool."""
    global db_pool
    if db_pool:
        await db_pool.close()
        # logger.info("Database connection pool closed")


async def search_knowledge_base(ctx: RunContext[None], query: str, limit: int = 5) -> str:
    """
    Search the knowledge base using semantic similarity.

    Args:
        query: The search query to find relevant information
        limit: Maximum number of results to return (default: 5)

    Returns:
        Formatted search results with source citations
    """
    try:
        # Ensure database is initialized
        if not db_pool:
            await initialize_db()

        # Generate embedding for query
        from ingestion.embedder import create_embedder
        embedder = create_embedder()
        query_embedding = await embedder.embed_query(query)

        # Convert to PostgreSQL vector format
        embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'

        # Hybrid retrieval: semantic + lexical candidate generation, then score fusion.
        keywords = extract_keywords(query)
        candidate_limit = max(30, limit * 8)
        keyword_limit = max(20, limit * 6)
        rerank_limit = max(12, limit * 4)

        async with db_pool.acquire() as conn:
            results = await conn.fetch(
                """
                WITH scored AS (
                    SELECT
                        c.id,
                        c.document_id,
                        c.chunk_index,
                        c.content,
                        c.metadata,
                        d.title AS document_title,
                        d.source AS document_source,
                        1 - (c.embedding <=> $1::vector) AS semantic_score,
                        (
                            SELECT COUNT(*)
                            FROM unnest($3::text[]) kw
                            WHERE c.content ILIKE ('%%' || kw || '%%')
                        )::int AS keyword_hits
                    FROM chunks c
                    JOIN documents d ON d.id = c.document_id
                    WHERE c.embedding IS NOT NULL
                ),
                semantic_candidates AS (
                    SELECT * FROM scored
                    ORDER BY semantic_score DESC
                    LIMIT $2
                ),
                lexical_candidates AS (
                    SELECT * FROM scored
                    WHERE keyword_hits > 0
                    ORDER BY keyword_hits DESC, semantic_score DESC
                    LIMIT $4
                ),
                merged AS (
                    SELECT * FROM semantic_candidates
                    UNION
                    SELECT * FROM lexical_candidates
                )
                SELECT
                    id AS chunk_id,
                    document_id,
                    chunk_index,
                    content,
                    metadata,
                    document_title,
                    document_source,
                    keyword_hits,
                    (
                        (semantic_score * 0.82) +
                        (LEAST(keyword_hits, 5)::float / 5.0 * 0.18)
                    ) AS similarity
                FROM merged
                ORDER BY similarity DESC
                LIMIT $5
                """,
                embedding_str,
                candidate_limit,
                keywords,
                keyword_limit,
                rerank_limit
            )

            expanded_results = list(results)
            for seed in list(results)[: min(8, len(results))]:
                seed_content = seed["content"] or ""
                seed_is_headerish = len(seed_content.strip()) <= 450 or seed_content.strip().endswith(":")
                if not seed_is_headerish:
                    continue

                neighbor = await conn.fetchrow(
                    """
                    SELECT
                        c.id AS chunk_id,
                        c.document_id,
                        c.chunk_index,
                        c.content,
                        c.metadata,
                        d.title AS document_title,
                        d.source AS document_source,
                        0::int AS keyword_hits,
                        $3::float AS similarity
                    FROM chunks c
                    JOIN documents d ON d.id = c.document_id
                    WHERE c.document_id = $1::uuid
                      AND c.chunk_index = $2
                    """,
                    seed["document_id"],
                    int(seed["chunk_index"]) + 1,
                    float(seed["similarity"]) * 0.985,
                )
                if neighbor:
                    expanded_results.append(neighbor)

            results = sorted(expanded_results, key=lambda r: float(r["similarity"]), reverse=True)

        # De-duplicate near-identical text blocks to improve answer grounding.
        deduped_results = []
        seen = set()
        for row in results:
            key = normalize_for_dedupe(row["content"])[:300]
            if not key or key in seen:
                continue
            seen.add(key)
            deduped_results.append(row)
            if len(deduped_results) >= max(1, limit):
                break

        if not deduped_results:
            return "No relevant information found in the knowledge base for your query."

        # Build response with sources
        response_parts = []
        for i, row in enumerate(deduped_results, 1):
            similarity = row['similarity']
            content = row['content']
            doc_title = row['document_title']
            doc_source = row['document_source']

            response_parts.append(
                f"[Source: {doc_title}]\n{content}\n"
            )

        if not response_parts:
            return "Found some results but they may not be directly relevant to your query. Please try rephrasing your question."

        return f"Found {len(response_parts)} relevant results:\n\n" + "\n---\n".join(response_parts)

    except Exception as e:
        # logger.error(f"Knowledge base search failed: {e}", exc_info=True)
        return f"I encountered an error searching the knowledge base: {str(e)}"


# Create the PydanticAI agent with the RAG tool
agent = Agent(
    resolve_agent_model(os.getenv("LLM_MODEL", "openai:gpt-4o-mini")),
    system_prompt="""You are an intelligent knowledge assistant with access to an organization's documentation and information.
Your role is to help users find accurate information from the knowledge base.
You have a professional yet friendly demeanor.

IMPORTANT: Always search the knowledge base before answering questions about specific information.
If search results are returned, answer using those results and do not switch to generic advice.
If information isn't in the knowledge base, clearly state that.
Be concise but thorough in your responses.
Ask clarifying questions if the user's query is ambiguous.
When you find relevant information, synthesize it clearly and cite the source documents.""",
    tools=[search_knowledge_base]
)


class RAGAgentCLI:
    """Enhanced CLI for interacting with the RAG Agent."""

    def __init__(self, force_rag: bool = False, rag_limit: int = 5):
        """Initialize CLI."""
        self.message_history = []
        self.force_rag = force_rag
        self.rag_limit = rag_limit

    async def _build_forced_rag_prompt(self, user_message: str) -> str:
        """
        Build an augmented prompt that always includes retrieved context.

        This guarantees retrieval happens for each user turn when force_rag is enabled.
        """
        retrieval = await search_knowledge_base(None, user_message, self.rag_limit)
        return (
            "You must answer using only the retrieved knowledge-base context below.\n"
            "Do not say the context is fragmented. Extract the best available explicit questions/items.\n"
            "If the exact answer is missing, say exactly that and stop. Do not provide generic advice.\n\n"
            f"User question:\n{user_message}\n\n"
            f"Retrieved context:\n{retrieval}\n"
        )

    def print_banner(self):
        """Print welcome banner."""
        print(f"\n{Colors.CYAN}{Colors.BOLD}{'=' * 60}")
        print("🤖 Docling RAG Knowledge Assistant")
        print("=" * 60)
        print(f"{Colors.WHITE}AI-powered document search with streaming responses")
        print(f"Type 'exit', 'quit', or Ctrl+C to exit")
        print(f"Type 'help' for commands")
        print("=" * 60 + f"{Colors.END}\n")

    def print_help(self):
        """Print help information."""
        help_text = f"""
{Colors.BOLD}Available Commands:{Colors.END}
  {Colors.GREEN}help{Colors.END}           - Show this help message
  {Colors.GREEN}clear{Colors.END}          - Clear conversation history
  {Colors.GREEN}stats{Colors.END}          - Show conversation statistics
  {Colors.GREEN}exit/quit{Colors.END}      - Exit the CLI

{Colors.BOLD}Usage:{Colors.END}
  Simply type your question and press Enter to chat with the agent.
  The agent will search the knowledge base and provide answers with source citations.

{Colors.BOLD}Features:{Colors.END}
  • Semantic search through embedded documents
  • Streaming responses in real-time
  • Conversation history maintained across turns
  • Source citations for all information

{Colors.BOLD}Examples:{Colors.END}
  - "What are the main topics in the knowledge base?"
  - "Tell me about [specific topic from your documents]"
  - "Summarize information about [subject]"
"""
        print(help_text)

    def print_stats(self):
        """Print conversation statistics."""
        message_count = len(self.message_history)
        print(f"\n{Colors.MAGENTA}{Colors.BOLD}📊 Session Statistics:{Colors.END}")
        print(f"  Messages in history: {message_count}")
        print(f"  Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{Colors.BLUE}{'─' * 60}{Colors.END}\n")

    async def check_database(self) -> bool:
        """Check database connection."""
        try:
            await initialize_db()
            async with db_pool.acquire() as conn:
                result = await conn.fetchval("SELECT 1")
                if result == 1:
                    print(f"{Colors.GREEN}✓ Database connection successful{Colors.END}")

                    # Check for documents
                    doc_count = await conn.fetchval("SELECT COUNT(*) FROM documents")
                    chunk_count = await conn.fetchval("SELECT COUNT(*) FROM chunks")

                    print(f"{Colors.GREEN}✓ Knowledge base ready: {doc_count} documents, {chunk_count} chunks{Colors.END}")
                    return True
            return False
        except Exception as e:
            print(f"{Colors.RED}✗ Database connection failed: {e}{Colors.END}")
            return False

    def extract_tool_calls(self, messages: List[Any]) -> List[Dict[str, Any]]:
        """Extract tool call information from messages."""
        from pydantic_ai.messages import ModelResponse, ToolCallPart

        tools_used = []
        for msg in messages:
            if isinstance(msg, ModelResponse):
                for part in msg.parts:
                    if isinstance(part, ToolCallPart):
                        tools_used.append({
                            'tool_name': part.tool_name,
                            'args': part.args,
                            'tool_call_id': part.tool_call_id
                        })
        return tools_used

    def format_tools_used(self, tools: List[Dict[str, Any]]) -> str:
        """Format tools used for display."""
        if not tools:
            return ""

        formatted = f"\n{Colors.MAGENTA}{Colors.BOLD}🛠 Tools Used:{Colors.END}\n"
        for i, tool in enumerate(tools, 1):
            tool_name = tool.get('tool_name', 'unknown')
            args = tool.get('args', {})

            formatted += f"  {Colors.CYAN}{i}. {tool_name}{Colors.END}"

            # Show key arguments for context (handle both dict and other types)
            if args and isinstance(args, dict):
                key_args = []
                if 'query' in args:
                    query_preview = str(args['query'])[:50] + '...' if len(str(args['query'])) > 50 else str(args['query'])
                    key_args.append(f"query='{query_preview}'")
                if 'limit' in args:
                    key_args.append(f"limit={args['limit']}")

                if key_args:
                    formatted += f" ({', '.join(key_args)})"

            formatted += "\n"

        return formatted

    async def stream_chat(self, message: str) -> None:
        """Send message to agent and display streaming response."""
        try:
            print(f"\n{Colors.BOLD}🤖 Assistant:{Colors.END} ", end="", flush=True)

            prompt = message
            if self.force_rag:
                prompt = await self._build_forced_rag_prompt(message)

            # Stream the response using run_stream
            async with agent.run_stream(
                prompt,
                message_history=self.message_history
            ) as result:
                # Stream text as it comes in (delta=True for only new tokens)
                async for text in result.stream_text(delta=True):
                    # Print only the new token
                    print(text, end="", flush=True)

                print()  # New line after streaming completes

                # Update message history for context
                self.message_history = result.all_messages()

                # Extract and display tools used in this turn
                new_messages = result.new_messages()
                tools_used = self.extract_tool_calls(new_messages)
                if tools_used:
                    print(self.format_tools_used(tools_used))
                elif self.force_rag:
                    print(f"\n{Colors.MAGENTA}{Colors.BOLD}🛠 Tools Used:{Colors.END}\n  {Colors.CYAN}1. forced_rag_retrieval{Colors.END}")

            # Print separator
            print(f"{Colors.BLUE}{'─' * 60}{Colors.END}")

        except Exception as e:
            print(f"\n{Colors.RED}✗ Error: {e}{Colors.END}")
            # logger.error(f"Chat error: {e}", exc_info=True)

    async def run(self):
        """Run the CLI main loop."""
        self.print_banner()

        # Check database connection
        if not await self.check_database():
            print(f"{Colors.RED}Cannot connect to database. Please check your DATABASE_URL.{Colors.END}")
            return

        print(f"{Colors.GREEN}Ready to chat! Ask me anything about the knowledge base.{Colors.END}\n")

        try:
            while True:
                try:
                    # Get user input
                    user_input = input(f"{Colors.BOLD}You: {Colors.END}").strip()

                    if not user_input:
                        continue

                    # Handle commands
                    if user_input.lower() in ['exit', 'quit', 'bye']:
                        print(f"{Colors.CYAN}👋 Thank you for using the knowledge assistant. Goodbye!{Colors.END}")
                        break
                    elif user_input.lower() == 'help':
                        self.print_help()
                        continue
                    elif user_input.lower() == 'clear':
                        self.message_history = []
                        print(f"{Colors.GREEN}✓ Conversation history cleared{Colors.END}")
                        continue
                    elif user_input.lower() == 'stats':
                        self.print_stats()
                        continue

                    # Send message to agent
                    await self.stream_chat(user_input)

                except KeyboardInterrupt:
                    print(f"\n{Colors.CYAN}👋 Goodbye!{Colors.END}")
                    break
                except EOFError:
                    print(f"\n{Colors.CYAN}👋 Goodbye!{Colors.END}")
                    break

        except Exception as e:
            print(f"{Colors.RED}✗ CLI error: {e}{Colors.END}")
            # logger.error(f"CLI error: {e}", exc_info=True)
        finally:
            await close_db()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Enhanced CLI for Docling RAG Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging (shows httpx and other debug logs)'
    )

    parser.add_argument(
        '--model',
        default=None,
        help='Override LLM model (e.g., gpt-4o)'
    )
    parser.add_argument(
        '--force-rag',
        action='store_true',
        help='Force retrieval every turn by injecting retrieved context before generation'
    )
    parser.add_argument(
        '--rag-limit',
        type=int,
        default=5,
        help='Number of chunks to retrieve when --force-rag is enabled (default: 5)'
    )

    args = parser.parse_args()

    # Configure logging - suppress all logs by default unless --verbose
    if args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.WARNING  # Only show warnings and errors

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Explicitly suppress httpx logging unless verbose mode
    if not args.verbose:
        logging.getLogger('httpx').setLevel(logging.WARNING)
        logging.getLogger('httpcore').setLevel(logging.WARNING)
        logging.getLogger('openai').setLevel(logging.WARNING)

    # Override model if specified
    if args.model:
        global agent
        agent = Agent(
            resolve_agent_model(args.model),
            system_prompt=agent.system_prompt,
            tools=[search_knowledge_base]
        )
        # logger.info(f"Using model: {args.model}")

    # Check required environment variables
    if not os.getenv("DATABASE_URL"):
        print(f"{Colors.RED}✗ DATABASE_URL environment variable is required{Colors.END}")
        sys.exit(1)

    # Only enforce API key if not using a local model/base URL
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("OPENAI_BASE_URL") and "openai" in os.getenv("LLM_MODEL", "openai").lower():
        print(f"{Colors.RED}✗ OPENAI_API_KEY environment variable is required (unless using local AI){Colors.END}")
        sys.exit(1)

    # Create and run CLI
    rag_limit = max(1, min(args.rag_limit, 20))
    cli = RAGAgentCLI(force_rag=args.force_rag, rag_limit=rag_limit)

    try:
        asyncio.run(cli.run())
    except KeyboardInterrupt:
        print(f"\n{Colors.CYAN}👋 Goodbye!{Colors.END}")
    except Exception as e:
        print(f"{Colors.RED}✗ CLI startup error: {e}{Colors.END}")
        # logger.error(f"Startup error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
