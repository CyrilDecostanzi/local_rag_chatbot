"""Centralised settings, read once from environment / .env.

Fixes over the original:
- CHUNK_SIZE / CHUNK_OVERLAP are cast to int (were strings).
- LLM_PROVIDER is normalised to lowercase and validated.
- SYSTEM_PROMPT has a sane default (was None).
"""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _get(key: str, default: str) -> str:
    val = os.getenv(key)
    return val if val is not None else default


@dataclass
class Settings:
    # Embedding
    embedding_model: str = "all-MiniLM-L6-v2"
    # FAISS
    index_dir: str = "faiss_index"
    top_k: int = 3
    retrieval_multiplier: int = 3
    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 50
    # Re-ranking
    use_reranking: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # LLM
    llm_provider: str = "ollama"  # "ollama" | "openai"
    llm_model: str = "mistral"
    openai_model: str = "gpt-3.5-turbo"
    openai_api_key: str = ""
    system_prompt: str = "You are a helpful AI assistant."
    # Security
    allow_dangerous_deserialization: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        provider = _get("LLM_PROVIDER", "ollama").lower()
        if provider not in ("ollama", "openai"):
            raise ValueError(
                f"LLM_PROVIDER must be 'ollama' or 'openai', got '{provider}'"
            )
        return cls(
            embedding_model=_get("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2"),
            index_dir=_get("INDEX_DIR", "faiss_index"),
            top_k=int(_get("DEFAULT_TOP_K", "3")),
            retrieval_multiplier=int(_get("RETRIEVAL_MULTIPLIER", "3")),
            chunk_size=int(_get("CHUNK_SIZE", "500")),
            chunk_overlap=int(_get("CHUNK_OVERLAP", "50")),
            use_reranking=_get("USE_RERANKING", "true").lower() == "true",
            rerank_model=_get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            llm_provider=provider,
            llm_model=_get("LLM_MODEL", "mistral"),
            openai_model=_get("OPENAI_MODEL", "gpt-3.5-turbo"),
            openai_api_key=_get("OPENAI_API_KEY", ""),
            system_prompt=_get(
                "SYSTEM_PROMPT", "You are a helpful AI assistant."
            ),
            allow_dangerous_deserialization=_get(
                "ALLOW_DANGEROUS_DESERIALIZATION", "false"
            ).lower() == "true",
        )
