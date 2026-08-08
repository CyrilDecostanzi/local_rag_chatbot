"""Index documents into a FAISS store.

Usage:
    python -m rag.cli.index --data_path ./data
"""
import argparse
import logging
import time

from rag.config import Settings
from rag.indexer import build_index
from rag.loader import load_documents

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Index documents into FAISS")
    parser.add_argument(
        "--data_path", type=str, default="./data", help="Directory with documents"
    )
    args = parser.parse_args()

    settings = Settings.from_env()
    start = time.time()

    chunks = load_documents(
        args.data_path, settings.chunk_size, settings.chunk_overlap
    )
    build_index(chunks, settings.embedding_model, settings.index_dir)

    logging.info("Completed in %.2f seconds", time.time() - start)


if __name__ == "__main__":
    main()
