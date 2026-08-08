"""Build and persist the FAISS vector store."""
import logging
import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)


def build_index(chunks, embedding_model: str, index_dir: str):
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    os.makedirs(index_dir, exist_ok=True)
    vectorstore.save_local(index_dir)
    logger.info("FAISS index saved to %s", index_dir)
    return vectorstore
