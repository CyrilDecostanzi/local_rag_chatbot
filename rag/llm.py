"""LLM factory — returns a LangChain chat model for the configured provider."""
import logging

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from rag.config import Settings

logger = logging.getLogger(__name__)


def build_llm(settings: Settings):
    if settings.llm_provider == "ollama":
        return ChatOllama(model=settings.llm_model)

    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when using OpenAI provider")
        return ChatOpenAI(
            model=settings.openai_model, api_key=settings.openai_api_key
        )

    raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")
