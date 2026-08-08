"""RAG chain built with LangChain Expression Language (LCEL).

The chain is a plain Runnable: prompt -> llm -> StrOutputParser. It takes a
dict {"question": str, "history": list} and returns the answer string.
Conversation history is injected by the caller (see rag/cli/chat.py).
"""
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from rag.config import Settings
from rag.llm import build_llm
from rag.retriever import build_retriever

_CONTEXT_PLACEHOLDER = (
    "Answer the user's question using ONLY the provided context. If the context\n"
    "does not contain the answer, say you don't know — do not make things up.\n\n"
    "Context:\n{context}"
)


def build_chain(settings: Settings):
    retriever = build_retriever(settings)
    llm = build_llm(settings)

    # Build the system message by appending the fixed RAG instructions.
    # Use plain concatenation rather than .format() so that curly braces
    # in settings.system_prompt are never mis-interpreted as format tokens.
    system_content = settings.system_prompt + "\n\n" + _CONTEXT_PLACEHOLDER

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_content),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    def format_context(inputs: dict) -> dict:
        question = inputs["question"]
        history: list[BaseMessage] = inputs.get("history", [])
        docs = retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in docs)
        return {"context": context, "question": question, "history": history}

    chain = (
        format_context
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
