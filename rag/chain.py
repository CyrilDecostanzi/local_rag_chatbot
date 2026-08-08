"""RAG chain built with LangChain Expression Language (LCEL).

The chain takes a dict {"question": str, "messages": list} and returns the
answer string. ``messages`` should be a list of LangChain message objects
(SystemMessage, HumanMessage, AIMessage) that represent the system prompt,
conversation history, and the current user question. The retriever appends
a system-level context block so the model can ground its answer in the
indexed documents.
"""
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser

from rag.config import Settings
from rag.llm import build_llm
from rag.retriever import build_retriever

CONTEXT_TEMPLATE = """Use the following context to answer the question:

{context}"""


def build_chain(settings: Settings):
    retriever = build_retriever(settings)
    llm = build_llm(settings)

    def invoke_with_context(inputs: dict) -> str:
        question = inputs["question"]
        messages = list(inputs.get("messages", []))
        docs = retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in docs)
        # Insert the context system message before the last HumanMessage so
        # the message order is: system, history, context, human — which is
        # expected by most chat model providers.
        messages.insert(-1, SystemMessage(content=CONTEXT_TEMPLATE.format(context=context)))
        return StrOutputParser().invoke(llm.invoke(messages))

    return invoke_with_context
