"""RAG chain built with LangChain Expression Language (LCEL).

The chain is a plain Runnable: prompt -> llm -> StrOutputParser. It takes a
dict {"question": ...} and returns the answer string. Conversation memory is
handled at the CLI layer (see cli.py) by injecting history into the prompt.
"""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from rag.config import Settings
from rag.llm import build_llm
from rag.retriever import build_retriever

SYSTEM_TEMPLATE = """You are a helpful AI assistant. Answer the user's question
using ONLY the provided context. If the context does not contain the answer,
say you don't know — do not make things up.

Context:
{context}
"""


def build_chain(settings: Settings):
    retriever = build_retriever(settings)
    llm = build_llm(settings)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_TEMPLATE),
            ("human", "{question}"),
        ]
    )

    def format_context(inputs: dict) -> dict:
        question = inputs["question"]
        docs = retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in docs)
        return {"context": context, "question": question}

    chain = (
        format_context
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
