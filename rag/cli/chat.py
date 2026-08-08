"""Interactive RAG chatbot with conversation memory.

Usage:
    python -m rag.cli.chat
"""
import logging

from langchain_core.messages import AIMessage, HumanMessage

from rag.chain import build_chain
from rag.config import Settings

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s"
)


def main() -> None:
    settings = Settings.from_env()
    chain = build_chain(settings)

    # Simple sliding-window conversation memory: keep the last N turns.
    history: list = []
    MAX_HISTORY = 6  # 3 exchanges

    logging.info(
        "Starting RAG chatbot with %s (%s)...",
        settings.llm_provider,
        settings.llm_model if settings.llm_provider == "ollama" else settings.openai_model,
    )

    while True:
        try:
            query = input("\n💬 Ask a question (or 'exit' to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if query.lower() == "exit":
            break

        try:
            answer = chain.invoke({"question": query, "history": history})
        except Exception:
            logging.exception("Error during chain invocation — skipping this turn.")
            continue

        print("\n🤖 Model's response:")
        print(answer)

        # Remember this turn (trim to keep the window bounded).
        history.append(HumanMessage(content=query))
        history.append(AIMessage(content=answer))
        if len(history) > MAX_HISTORY:
            history = history[-MAX_HISTORY:]


if __name__ == "__main__":
    main()
