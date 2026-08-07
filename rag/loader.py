"""Document loading + chunking.

Fixes over the original:
- Splits per document (split_documents) so a chunk never mixes two files.
- Attaches the source filename to each chunk's metadata.
- Raises a clear error if the data dir is missing or empty.
"""
import logging
import os

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def load_documents(data_path: str, chunk_size: int, chunk_overlap: int):
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    docs = []
    for filename in sorted(os.listdir(data_path)):
        path = os.path.join(data_path, filename)
        try:
            if filename.endswith(".txt"):
                loader = TextLoader(path, encoding="utf-8")
            elif filename.endswith(".pdf"):
                loader = PyPDFLoader(path)
            else:
                continue
            loaded = loader.load()
            for doc in loaded:
                doc.metadata["source"] = filename
            docs.extend(loaded)
            logger.info("Loaded %s (%d chunk(s))", filename, len(loaded))
        except Exception as e:  # noqa: BLE001 - keep indexing going on bad files
            logger.error("Error processing %s: %s", filename, e)

    if not docs:
        raise ValueError(f"No supported documents found in {data_path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    logger.info("Split into %d chunks", len(chunks))
    return chunks
