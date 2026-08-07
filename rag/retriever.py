"""Build the retriever, optionally wrapped in a cross-encoder re-ranker."""
import logging

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from rag.config import Settings

logger = logging.getLogger(__name__)


def build_retriever(settings: Settings):
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    vectorstore = FAISS.load_local(
        settings.index_dir,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    base_retriever = vectorstore.as_retriever(
        search_kwargs={"k": settings.top_k * settings.retrieval_multiplier}
    )

    if not settings.use_reranking:
        return base_retriever

    logger.info("Loading cross-encoder for re-ranking...")
    compressor = CrossEncoderReranker(
        model=HuggingFaceCrossEncoder(model_name=settings.rerank_model),
        top_n=settings.top_k,
    )
    return ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )
