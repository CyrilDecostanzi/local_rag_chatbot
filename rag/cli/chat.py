"""Interactive RAG chatbot with conversation memory.

Usage:
    python -m rag.cli.chat
"""
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

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

        # Build the message list: system + history + current question.
        messages = [SystemMessage(content=settings.system_prompt)]
        messages.extend(history)
        messages.append(HumanMessage(content=query))

        answer = chain.invoke({"question": query})
        print("\n🤖 Model's response:")
        print(answer)

        # Remember this turn (trim to keep the window bounded).
        history.append(HumanMessage(content=query))
        history.append(AIMessage(content=answer))
        if len(history) > MAX_HISTORY:
            history = history[-MAX_HISTORY:]


if __name__ == "__main__":
    main()
