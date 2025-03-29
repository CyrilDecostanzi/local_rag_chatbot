#!/usr/bin/env python3
"""
Retrieval module for the RAG chatbot.
This module handles retrieving relevant documents based on user queries.
"""
import logging
import operator
import os
import pickle

import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

from src.config.settings import (
    EMBEDDING_MODEL_NAME,
    INDEX_DIR,
    DEFAULT_TOP_K,
    RETRIEVAL_MULTIPLIER,
    USE_RERANKING,
    RERANK_MODEL,
)

# Log configuration
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

class DocumentRetriever:
    """Class for retrieving relevant documents for RAG"""
    
    def __init__(self):
        """Initialize the retriever with the embedding model and index"""
        self.index = None
        self.documents = None
        self.embed_model = None
        self.reranker = None
        self._load_models()
        self._load_index()
    
    def _load_models(self):
        """Load the embedding model and reranker"""
        logging.info(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        if USE_RERANKING:
            logging.info("Loading cross-encoder for re-ranking...")
            self.reranker = CrossEncoder(RERANK_MODEL)
    
    def _load_index(self):
        """Load the FAISS index and documents if they exist"""
        index_path = os.path.join(INDEX_DIR, "index.faiss")
        docs_path = os.path.join(INDEX_DIR, "documents.pkl")
        
        if os.path.exists(index_path) and os.path.exists(docs_path):
            try:
                logging.info(f"Loading FAISS index from {index_path}...")
                self.index = faiss.read_index(index_path)
                
                logging.info(f"Loading documents from {docs_path}...")
                with open(docs_path, "rb") as f:
                    self.documents = pickle.load(f)
                    
                logging.info(f"Successfully loaded {len(self.documents)} documents and index.")
            except Exception as e:
                logging.error(f"Error loading index or documents: {e}")
                self.index = None
                self.documents = None
        else:
            logging.warning(f"FAISS index not found at {index_path} or documents not found at {docs_path}.")
            logging.warning("Please index your documents first using 'python app.py --index' or through the GUI.")
    
    def retrieve_documents(self, query, top_k=DEFAULT_TOP_K, rerank=USE_RERANKING):  
        """
        Retrieve the most relevant documents for a query.
        
        Args:
            query (str): The user query
            top_k (int): Number of documents to retrieve
            rerank (bool): Whether to rerank the results with a cross-encoder
            
        Returns:
            list: List of retrieved document texts
        """
        # Check if index and documents are loaded
        if self.index is None or self.documents is None:
            logging.error("Cannot retrieve documents: No index or documents available.")
            return ["Erreur: L'index RAG n'existe pas. Veuillez indexer des documents avant de poser des questions."]
        
        # Retrieve the `top_k` closest documents to the query.
        # If rerank=True, re-order the results with the cross-encoder.
        query_embedding = self.embed_model.encode([query])
        D, I = self.index.search(query_embedding, top_k * RETRIEVAL_MULTIPLIER)
        candidates = [self.documents[i] for i in I[0] if 0 <= i < len(self.documents)]
        
        if rerank and self.reranker and candidates:
            logging.info("Re-ranking documents...")
            pairs = [(query, doc) for doc in candidates]
            
            # Batch processing for reranking
            batch_size = 32
            scores = []
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i + batch_size]
                batch_scores = self.reranker.predict(batch_pairs)
                scores.extend(batch_scores)
            
            # Select top_k
            ranked = sorted(zip(candidates, scores), key=operator.itemgetter(1), reverse=True)
            retrieved_texts = [doc for doc, score in ranked[:top_k]]
        else:
            retrieved_texts = candidates[:top_k]
        
        return retrieved_texts

# Create a singleton instance
retriever = DocumentRetriever()

def retrieve_documents(query, top_k=DEFAULT_TOP_K, rerank=USE_RERANKING):
    """
    Retrieve relevant documents for a query.
    This is a helper function that uses the DocumentRetriever class.
    
    Args:
        query (str): The user query
        top_k (int): Number of documents to retrieve
        rerank (bool): Whether to rerank the results
        
    Returns:
        list: List of retrieved document texts
    """
    return retriever.retrieve_documents(query, top_k, rerank)

# Command line test
if __name__ == "__main__":
    query = input("🔍 Ask a question: ")
    docs = retrieve_documents(query)
    
    print("\n📚 Relevant documents:")
    for i, doc in enumerate(docs, 1):
        print(f"\n📄 Document {i}:\n{doc[:500]}...\n") 