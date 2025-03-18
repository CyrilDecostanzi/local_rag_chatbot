#!/usr/bin/env python3
import os
import time
import pickle
import argparse
import logging

import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import os

load_dotenv()


# Log configuration
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
INDEX_DIR = os.getenv("INDEX_DIR", "faiss_index")
CHUNK_SIZE = os.getenv("CHUNK_SIZE", 500)
CHUNK_OVERLAP = os.getenv("CHUNK_OVERLAP", 50)

def load_and_process_files(data_path):
    # Load and split files (.txt, .pdf) from the `data_path` directory
    # into text chunks.
    all_texts = []
    for file in os.listdir(data_path):
        file_path = os.path.join(data_path, file)
        try:
            if file.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                    if text:
                        all_texts.append(text)
            elif file.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                pages = loader.load()
                for page in pages:
                    content = page.page_content.strip()
                    if content:
                        all_texts.append(content)
        except Exception as e:
            logging.error(f"Error processing file {file}: {e}")

    if not all_texts:
        logging.warning("No text found in the directory.")
        return []

    # Concatenate and split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    full_text = "\n\n".join(all_texts)
    chunks = splitter.split_text(full_text)
    logging.info(f"Split into {len(chunks)} chunks.")
    return chunks

def index_documents(texts, embed_model, index):
    # Encode and add texts to the FAISS index.
    if not texts:
        logging.error("No text to index.")
        return
    logging.info("Starting document indexing...")
    embeddings = embed_model.encode(texts, show_progress_bar=True)
    index.add(embeddings)
    logging.info(f"✅ {len(texts)} chunks indexed in FAISS.")

def save_faiss_index(index, documents, output_dir=INDEX_DIR):
    # Save the FAISS index and original texts.
    os.makedirs(output_dir, exist_ok=True)
    index_path = os.path.join(output_dir, "index.faiss")
    docs_path = os.path.join(output_dir, "documents.pkl")
    
    faiss.write_index(index, index_path)
    with open(docs_path, "wb") as f:
        pickle.dump(documents, f)
    logging.info(f"💾 FAISS index saved in {output_dir}.")

def main(args):
    start_time = time.time()
    
    # Initialize the embedding model
    logging.info(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    # Create the FAISS index with the dimension corresponding to the model
    embedding_dim = embed_model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatL2(embedding_dim)
    
    # Load and process files
    docs = load_and_process_files(args.data_path)
    if docs:
        index_documents(docs, embed_model, index)
        save_faiss_index(index, docs)
    else:
        logging.error("No document indexed.")
    
    elapsed = time.time() - start_time
    logging.info(f"Completed in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Document indexing with FAISS")
    parser.add_argument("--data_path", type=str, default="./data", help="Directory containing the documents")
    args = parser.parse_args()
    main(args)
