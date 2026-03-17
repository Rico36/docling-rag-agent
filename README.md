# Docling RAG Agent

> Attribution: This repository is largely based on work by Cole Medin (`coleam00`) from the Ottomator Agents project: https://github.com/coleam00/ottomator-agents/tree/main/docling-rag-agent

An intelligent text-based CLI agent that provides conversational access to a knowledge base stored in PostgreSQL with PGVector. Uses RAG (Retrieval Augmented Generation) to search through embedded documents and provide contextual, accurate responses with source citations. Supports multiple document formats including audio files with Whisper transcription.

## 🎓 New to Docling?

**Start with the tutorials!** Check out the [`docling_basics/`](./docling_basics/) folder for progressive examples that teach Docling fundamentals:

1. **Simple PDF Conversion** - Basic document processing
2. **Multiple Format Support** - PDF, Word, PowerPoint handling
3. **Audio Transcription** - Speech-to-text with Whisper
4. **Hybrid Chunking** - Intelligent chunking for RAG systems

These tutorials provide the foundation for understanding how this full RAG agent works. [**→ Go to Docling Basics**](./docling_basics/)

## Features

- 💬 Interactive text-based CLI with streaming responses
- 🔍 Semantic search through vector-embedded documents
- 📚 Context-aware responses using RAG pipeline
- 🎯 Source citation for all information provided
- 🔄 Real-time streaming text output as tokens arrive
- 💾 PostgreSQL/PGVector for scalable knowledge storage
- 🧠 Conversation history maintained across turns
- 🎙️ Audio transcription with Whisper ASR (MP3 files)

## Prerequisites

- Python 3.9 or later
- [uv](https://docs.astral.sh/uv/) (Recommended)
  - Windows: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
  - Alternative: `pip install uv`
  - macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- PostgreSQL with PGVector extension (Supabase, Neon, self-hosted Postgres, etc.)
- API Keys:
  - OpenAI API key (for embeddings and LLM)

### GPU Acceleration (Windows Users)

By default, PyTorch often installs the CPU-only version on Windows. To enable NVIDIA GPU support for faster document processing (Docling):

1. **Install PyTorch with CUDA support:**
   ```bash
   uv pip install --python .venv\Scripts\python.exe torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 --reinstall
   ```
   *Note: Adjust `cu126` (CUDA 12.6) to match your NVIDIA driver/toolkit compatibility.*

### Troubleshooting GPU

If you see `No GPU detected` in the logs even after configuration:

1.  **Reset the environment:** This is necessary to clear out any cached CPU-only versions of PyTorch.

    **Windows (PowerShell):**
    ```powershell
    # Remove locked CPU versions
    Remove-Item uv.lock -ErrorAction SilentlyContinue
    Remove-Item .venv -Recurse -Force -ErrorAction SilentlyContinue
    # Re-install from correct sources
    uv sync
    ```

    **macOS / Linux:**
   ```bash
   # Remove locked CPU versions
   rm -f uv.lock
   rm -rf .venv
   # Re-install from correct sources
   uv sync
   ```

2. **Verify the exact project environment has CUDA-enabled torch:**
   ```bash
   .venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
   ```

### Preflight Check (Recommended)

Before running ingestion, run the preflight script to verify CUDA, `.env` embedding settings, and database vector dimensions are aligned:

```powershell
powershell -ExecutionPolicy Bypass -File .\preflight.ps1 -RequireCuda
```

## Quick Start

### 1. Install Dependencies

```bash
# Install dependencies using UV
uv sync
```

### 2. Set Up Environment Variables

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Required variables:
- `DATABASE_URL` - PostgreSQL connection string with PGVector extension
  - Example: `postgresql://user:password@localhost:5432/dbname`
  - Supabase: `postgresql://postgres.[project-ref]:[password]@aws-0-[region].pooler.supabase.com:5432/postgres`
  - Neon: `postgresql://[user]:[password]@[endpoint].neon.tech/[dbname]`

- `OPENAI_API_KEY` - OpenAI API key for embeddings and LLM
  - Get from: https://platform.openai.com/api-keys

- `OPENAI_BASE_URL` - OpenAI API URL (optional if using standard OpenAI)
  - Default: `https://api.openai.com/v1`

Optional variables:
- `LLM_MODEL` - LLM model to use (default: `openai:gpt-4o-mini`)
- `EMBEDDING_MODEL` - Embedding model (default: `text-embedding-3-small`)

### 2b. (Optional) Setup Local AI (Ollama)

You have two options for local embeddings. **Choose only one.**

**Option A: Use Ollama for Everything (Recommended)**
Uses the model running inside Ollama.

1. **Pull models in Ollama:** `ollama pull llama3` and `ollama pull nomic-embed-text`
2. **Configure `.env`:**
   ```bash
   OPENAI_BASE_URL=http://localhost:11434/v1
   OPENAI_API_KEY=ollama
   LLM_MODEL=openai:llama3
   
   # Configure Embedding (Note: nomic-embed-text is 768 dimensions)
   EMBEDDING_MODEL=nomic-embed-text
   EMBEDDING_DIMENSIONS=768
   # Do NOT set EMBEDDING_BACKEND=local (leave it commented out or remove it)
   ```

**Option B: Python In-Process Embeddings**
Runs embeddings directly in Python (faster, but downloads weights from Hugging Face).

1. **Install:**
   ```bash
   pip install sentence-transformers
   ```

2. **Configure `.env`:**
   ```bash
   # Point to local Ollama instance (ensure it's running)
   OPENAI_BASE_URL=http://localhost:11434/v1
   OPENAI_API_KEY=ollama
   LLM_MODEL=openai:llama3.2:3b
   
   EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1.5
   EMBEDDING_BACKEND=local
   ```

### 3. Configure Database

You must set up your PostgreSQL database with the PGVector extension and create the required schema:

1. **Enable PGVector extension** in your database (most cloud providers have this pre-installed)
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

2. **Run the schema file** to create tables and functions:
   *Note: If using local embeddings with different dimensions (e.g., 768), you must edit `sql/schema.sql` to match `vector(768)` before running this.*
   ```bash
   # In the SQL editor in Supabase/Neon, run:
   sql/schema.sql

   # Or using psql
   psql $DATABASE_URL < sql/schema.sql
   ```

The schema file (`sql/schema.sql`) creates:
- `documents` table for storing original documents with metadata
- `chunks` table for text chunks with 1536-dimensional embeddings
- `match_chunks()` function for vector similarity search

### 4. Ingest Documents

Add your documents to the `documents/` folder. **Multiple formats supported via Docling**:

**Supported Formats:**
- 📄 **PDF** (`.pdf`)
- 📝 **Word** (`.docx`, `.doc`)
- 📊 **PowerPoint** (`.pptx`, `.ppt`)
- 📈 **Excel** (`.xlsx`, `.xls`)
- 🌐 **HTML** (`.html`, `.htm`)
- 📋 **Markdown** (`.md`, `.markdown`)
- 📃 **Text** (`.txt`)
- 🎵 **Audio** (`.mp3`) - transcribed with Whisper

```bash
# Ingest all supported documents in the documents/ folder
# NOTE: By default, this CLEARS existing data before ingestion
uv run python -m ingestion.ingest --documents documents/

# Adjust chunk size (default: 1000)
uv run python -m ingestion.ingest --documents documents/ --chunk-size 800
```

**⚠️ Important:** The ingestion process **automatically deletes all existing documents and chunks** from the database before adding new documents. This ensures a clean state and prevents duplicate data.

The ingestion pipeline will:
1. **Auto-detect file type** and use Docling for PDFs, Office docs, HTML, and audio
2. **Transcribe audio files** using Whisper Turbo ASR with timestamps
3. **Convert to Markdown** for consistent processing
4. **Split into semantic chunks** with configurable size
5. **Generate embeddings** using OpenAI
6. **Store in PostgreSQL** with PGVector for similarity search

### 5. Run the Agent

```bash
# Run the Docling RAG Agent CLI
uv run python cli.py
```

**Features:**
- 🎨 **Colored output** for better readability
- 📊 **Session statistics** (`stats` command)
- 🔄 **Clear history** (`clear` command)
- 💡 **Built-in help** (`help` command)
- ✅ **Database health check** on startup
- 🔍 **Real-time streaming** responses

**Available commands:**
- `help` - Show help information
- `clear` - Clear conversation history
- `stats` - Show session statistics
- `exit` or `quit` - Exit the CLI

**Example interaction:**
```
============================================================
🤖 Docling RAG Knowledge Assistant
============================================================
AI-powered document search with streaming responses
Type 'exit', 'quit', or Ctrl+C to exit
Type 'help' for commands
============================================================

✓ Database connection successful
✓ Knowledge base ready: 20 documents, 156 chunks
Ready to chat! Ask me anything about the knowledge base.

You: What topics are covered in the knowledge base?
🤖 Assistant: Based on the knowledge base, the main topics include...

────────────────────────────────────────────────────────────
You: quit
👋 Thank you for using the knowledge assistant. Goodbye!
```

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   CLI User  │───▶│  RAG Agent   │────▶│ PostgreSQL  │
│   (Input)   │     │ (PydanticAI) │     │  PGVector   │
└─────────────┘     └──────────────┘     └─────────────┘
                           │
                    ┌──────┴──────┐
                    │             │
              ┌─────▼────┐  ┌────▼─────┐
              │  OpenAI  │  │  OpenAI  │
              │   LLM    │  │Embeddings│
              └──────────┘  └──────────┘
```

## Audio Transcription Feature

Audio files are automatically transcribed using **OpenAI Whisper Turbo** model:

**How it works:**
1. When ingesting audio files (MP3 supported currently), Docling uses Whisper ASR
2. Whisper generates accurate transcriptions with timestamps
3. Transcripts are formatted as markdown with time markers
4. Audio content becomes fully searchable through the RAG system

**Benefits:**
- 🎙️ **Speech-to-text**: Convert podcasts, interviews, lectures into searchable text
- ⏱️ **Timestamps**: Track when specific content was mentioned
- 🔍 **Semantic search**: Find audio content by topic or keywords
- 🤖 **Fully automatic**: Drop audio files in `documents/` folder and run ingestion

**Model details:**
- Model: `openai/whisper-large-v3-turbo`
- Optimized for: Speed and accuracy balance
- Languages: Multilingual support (90+ languages)
- Output format: Markdown with timestamps like `[time: 0.0-4.0] Transcribed text here`

**Example transcript format:**
```markdown
[time: 0.0-4.0] Welcome to our podcast on AI and machine learning.
[time: 5.28-9.96] Today we'll discuss retrieval augmented generation systems.
```

## Key Components

### RAG Agent

The main agent (`rag_agent.py`) that:
- Manages database connections with connection pooling
- Handles interactive CLI with streaming responses
- Performs knowledge base searches via RAG
- Tracks conversation history for context

### search_knowledge_base Tool

Function tool registered with the agent that:
- Generates query embeddings using OpenAI
- Searches using PGVector cosine similarity
- Returns top-k most relevant chunks
- Formats results with source citations

Example tool definition:
```python
async def search_knowledge_base(
    ctx: RunContext[None],
    query: str,
    limit: int = 5
) -> str:
    """Search the knowledge base using semantic similarity."""
    # Generate embedding for query
    # Search PostgreSQL with PGVector
    # Format and return results
```

### Database Schema

- `documents`: Stores original documents with metadata
  - `id`, `title`, `source`, `content`, `metadata`, `created_at`, `updated_at`

- `chunks`: Stores text chunks with vector embeddings
  - `id`, `document_id`, `content`, `embedding` (vector(1536)), `chunk_index`, `metadata`, `token_count`

- `match_chunks()`: PostgreSQL function for vector similarity search
  - Uses cosine similarity (`1 - (embedding <=> query_embedding)`)
  - Returns chunks with similarity scores above threshold

## Performance Optimization

### Database Connection Pooling
```python
db_pool = await asyncpg.create_pool(
    DATABASE_URL,
    min_size=2,
    max_size=10,
    command_timeout=60
)
```

### Embedding Cache
The embedder includes built-in caching for frequently searched queries, reducing API calls and latency.

### Streaming Responses
Token-by-token streaming provides immediate feedback to users while the LLM generates responses:
```python
async with agent.run_stream(user_input, message_history=history) as result:
    async for text in result.stream_text(delta=False):
        print(f"\rAssistant: {text}", end="", flush=True)
```

## Docker Deployment

### Using Docker Compose

```bash
# Start all services
docker-compose up -d

# Ingest documents
docker-compose --profile ingestion up ingestion

# View logs
docker-compose logs -f rag-agent
```

## API Reference

### search_knowledge_base Tool

```python
async def search_knowledge_base(
    ctx: RunContext[None],
    query: str,
    limit: int = 5
) -> str:
    """
    Search the knowledge base using semantic similarity.

    Args:
        query: The search query to find relevant information
        limit: Maximum number of results to return (default: 5)

    Returns:
        Formatted search results with source citations
    """
```

### Database Functions

```sql
-- Vector similarity search
SELECT * FROM match_chunks(
    query_embedding::vector(1536),
    match_count INT,
    similarity_threshold FLOAT DEFAULT 0.7
)
```

Returns chunks with:
- `id`: Chunk UUID
- `content`: Text content
- `embedding`: Vector embedding
- `similarity`: Cosine similarity score (0-1)
- `document_title`: Source document title
- `document_source`: Source document path

## Project Structure

```
docling-rag-agent/
├── cli.py                   # Enhanced CLI with colors and features (recommended)
├── rag_agent.py             # Basic CLI agent with PydanticAI
├── ingestion/
│   ├── ingest.py            # Document ingestion pipeline
│   ├── embedder.py          # Embedding generation with caching
│   └── chunker.py           # Document chunking logic
├── utils/
│   ├── providers.py         # OpenAI model/client configuration
│   ├── db_utils.py          # Database connection pooling
│   └── models.py            # Pydantic models for config
├── sql/
│   └── schema.sql           # PostgreSQL schema with PGVector
├── documents/               # Sample documents for ingestion
├── pyproject.toml           # Project dependencies
├── .env.example             # Environment variables template
└── README.md                # This file
```
