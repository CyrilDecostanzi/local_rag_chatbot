# 📚 Local RAG Chatbot with LangChain, FAISS and Ollama (or OpenAI)

A Retrieval-Augmented Generation (RAG) chatbot that runs locally. It uses
**LangChain** (LCEL) to build the pipeline, **FAISS** for semantic retrieval,
and either **Ollama** (local models) or **OpenAI** (cloud) for generation.

## How it works

1. **Indexing** — documents (`.txt`, `.pdf`) are split into chunks and embedded
   with Sentence Transformers, then stored in a FAISS index.
2. **Retrieval** — a query is embedded and matched against the index; results
   are optionally re-ranked with a cross-encoder.
3. **Generation** — the retrieved context is fed to an LLM (Ollama or OpenAI)
   which answers the question.

## Features

- **LangChain LCEL** pipeline (prompt → LLM → output parser)
- **Local or cloud LLM**: Ollama or OpenAI
- **Semantic retrieval** with FAISS
- **Optional cross-encoder re-ranking**
- **Conversation memory** (sliding window)
- **Per-document chunking** with source metadata
- Supports `.txt` and `.pdf`

## Installation

```bash
git clone https://github.com/CyrilDecostanzi/local_rag_chatbot.git
cd local_rag_chatbot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### LLM provider

**Option A — Ollama (local):**
```bash
ollama pull mistral
```

**Option B — OpenAI (cloud):** set `OPENAI_API_KEY` in `.env`.

## Configuration

Copy `.env.example` to `.env` and adjust:

```bash
cp .env.example .env
```

Key settings: `LLM_PROVIDER` (`ollama` or `openai`), `LLM_MODEL`, `OPENAI_MODEL`,
`EMBEDDING_MODEL_NAME`, `DEFAULT_TOP_K`, `USE_RERANKING`.

## Indexing documents

Place documents in `./data`, then:

```bash
python -m rag.cli.index --data_path ./data
```

## Chat

```bash
python -m rag.cli.chat
```

```
💬 Ask a question (or 'exit' to quit): How does RAG improve query accuracy?
```

## Project layout

```
rag/
├── config.py      # Settings from .env (validated, typed)
├── loader.py      # Document loading + per-document chunking
├── indexer.py     # Build & persist FAISS index
├── retriever.py   # Retriever + optional cross-encoder re-ranker
├── llm.py         # LLM factory (ollama / openai)
├── chain.py       # LCEL RAG chain
└── cli/
    ├── index.py   # python -m rag.cli.index
    └── chat.py    # python -m rag.cli.chat
```

## License

MIT — see [LICENSE](LICENSE).
