#!/usr/bin/env python3
"""
Indexing module for the RAG chatbot.
This module handles indexing documents for the RAG system.
"""
import os
import time
import pickle
import logging
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from src.config.settings import (
    EMBEDDING_MODEL_NAME,
    INDEX_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

# Log configuration
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

class DocumentIndexer:
    """Class for indexing documents for the RAG system"""
    
    def __init__(self):
        """Initialize the indexer with the embedding model"""
        self.embed_model = None
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model"""
        logging.info(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    def load_and_process_files(self, data_path):
        """
        Load and split files (.txt, .pdf) from the data_path directory into text chunks.
        
        Args:
            data_path (str): Path to the directory containing documents
            
        Returns:
            list: List of text chunks
        """
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
    
    def index_documents(self, texts):
        """
        Encode and add texts to a FAISS index.
        
        Args:
            texts (list): List of text chunks to index
            
        Returns:
            faiss.Index: The FAISS index with the indexed documents
        """
        if not texts:
            logging.error("No text to index.")
            return None
            
        logging.info("Starting document indexing...")
        
        # Create the FAISS index with the dimension corresponding to the model
        embedding_dim = self.embed_model.get_sentence_embedding_dimension()
        index = faiss.IndexFlatL2(embedding_dim)
        
        # Encode and add to index
        embeddings = self.embed_model.encode(texts, show_progress_bar=True)
        index.add(embeddings)
        logging.info(f"✅ {len(texts)} chunks indexed in FAISS.")
        
        return index
    
    def save_faiss_index(self, index, documents, output_dir=INDEX_DIR):
        """
        Save the FAISS index and original texts.
        
        Args:
            index (faiss.Index): The FAISS index to save
            documents (list): The original text chunks
            output_dir (str): Directory to save the index and documents
        """
        os.makedirs(output_dir, exist_ok=True)
        index_path = os.path.join(output_dir, "index.faiss")
        docs_path = os.path.join(output_dir, "documents.pkl")
        
        faiss.write_index(index, index_path)
        with open(docs_path, "wb") as f:
            pickle.dump(documents, f)
        logging.info(f"💾 FAISS index saved in {output_dir}.")
    
    def run_indexing(self, data_path):
        """
        Run the complete indexing process.
        
        Args:
            data_path (str): Path to the directory containing documents
            
        Returns:
            bool: True if indexing was successful, False otherwise
        """
        start_time = time.time()
        
        # Load and process files
        docs = self.load_and_process_files(data_path)
        if not docs:
            logging.error("No document indexed.")
            return False
            
        # Index documents
        index = self.index_documents(docs)
        if index is None:
            return False
            
        # Save index
        self.save_faiss_index(index, docs)
        
        elapsed = time.time() - start_time
        logging.info(f"Completed in {elapsed:.2f} seconds.")
        
        return True

# Create a singleton instance
indexer = DocumentIndexer()

def run_indexing(data_path):
    """
    Helper function to run the complete indexing process.
    
    Args:
        data_path (str): Path to the directory containing documents
        
    Returns:
        bool: True if indexing was successful, False otherwise
    """
    return indexer.run_indexing(data_path)

# Command line test
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Document indexing with FAISS")
    parser.add_argument("--data_path", type=str, default="./data", help="Directory containing the documents")
    args = parser.parse_args()
    run_indexing(args.data_path) 