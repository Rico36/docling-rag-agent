"""
RAG CLI Agent with PostgreSQL/PGVector
=======================================
Text-based CLI agent that searches through knowledge base using semantic similarity
"""

import asyncio
import asyncpg
import argparse
import json
import logging
import os
import re
import sys
from typing import Any

from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext

# Load environment variables
load_dotenv(".env")

logger = logging.getLogger(__name__)

# Global database pool
db_pool = None
ARGS = argparse.Namespace(force_rag=False, rag_limit=5)
STOP_WORDS = {
    "the", "and", "for", "with", "from", "that", "this", "what", "when", "where", "which",
    "should", "would", "could", "about", "into", "during", "phase", "list", "only", "your",
    "have", "has", "are", "was", "were", "why", "how", "who", "whom", "their", "them",
    "then", "than", "also", "any", "all", "can", "may", "our", "you", "use", "using",
}


def resolve_agent_model(raw_model: str) -> str:
    """Normalize model string for PydanticAI provider:model format."""
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


def extract_keywords(query: str, max_keywords: int = 10) -> list[str]:
    """Extract simple lexical keywords for hybrid retrieval."""
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", (query or "").lower())
    keywords: list[str] = []
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
        logger.info("Database connection pool initialized")


async def close_db():
    """Close database connection pool."""
    global db_pool
    if db_pool:
        await db_pool.close()
        logger.info("Database connection pool closed")


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
        logger.error(f"Knowledge base search failed: {e}", exc_info=True)
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


async def run_cli():
    """Run the agent in an interactive CLI with streaming."""

    # Initialize database
    await initialize_db()

    print("=" * 60)
    print("RAG Knowledge Assistant")
    print("=" * 60)
    print("Ask me anything about the knowledge base!")
    print("Type 'quit', 'exit', or press Ctrl+C to exit.")
    print("=" * 60)
    print()

    message_history = []

    try:
        while True:
            # Get user input
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nAssistant: Thank you for using the knowledge assistant. Goodbye!")
                break

            print("Assistant: ", end="", flush=True)

            try:
                prompt = user_input
                if ARGS.force_rag:
                    retrieval = await search_knowledge_base(None, user_input, ARGS.rag_limit)
                    prompt = (
                        "You must answer using only the retrieved knowledge-base context below.\n"
                        "Do not say the context is fragmented. Extract the best available explicit questions/items.\n"
                        "If the exact answer is missing, say exactly that and stop. Do not provide generic advice.\n\n"
                        f"User question:\n{user_input}\n\n"
                        f"Retrieved context:\n{retrieval}\n"
                    )

                # Stream the response using run_stream
                async with agent.run_stream(
                    prompt,
                    message_history=message_history
                ) as result:
                    # Stream text as it comes in (delta=True for only new tokens)
                    async for text in result.stream_text(delta=True):
                        # Print only the new token
                        print(text, end="", flush=True)

                    print()  # New line after streaming completes

                    # Update message history for context
                    message_history = result.all_messages()

            except KeyboardInterrupt:
                print("\n\n[Interrupted]")
                break
            except Exception as e:
                print(f"\n\nError: {e}")
                logger.error(f"Agent error: {e}", exc_info=True)

            print()  # Extra line for readability

    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    finally:
        await close_db()


async def main():
    """Main entry point."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Check required environment variables
    if not os.getenv("DATABASE_URL"):
        logger.error("DATABASE_URL environment variable is required")
        sys.exit(1)

    if not os.getenv("OPENAI_API_KEY") and not os.getenv("OPENAI_BASE_URL") and "openai" in os.getenv("LLM_MODEL", "openai").lower():
        logger.error("OPENAI_API_KEY environment variable is required (unless using local AI)")
        sys.exit(1)

    # Run the CLI
    await run_cli()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basic RAG CLI agent")
    parser.add_argument(
        "--force-rag",
        action="store_true",
        help="Force retrieval every turn by injecting retrieved context before generation",
    )
    parser.add_argument(
        "--rag-limit",
        type=int,
        default=5,
        help="Number of chunks to retrieve when --force-rag is enabled (default: 5)",
    )
    ARGS = parser.parse_args()
    ARGS.rag_limit = max(1, min(ARGS.rag_limit, 20))
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nShutting down...")
