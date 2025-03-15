#!/usr/bin/env python3
import os
import pickle
import logging

import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

# Log configuration
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

# Parameters
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_DIR = "faiss_index"
DEFAULT_TOP_K = 3

# Loading the embedding model
logging.info(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Loading the FAISS index
index_path = os.path.join(INDEX_DIR, "index.faiss")
logging.info(f"Loading FAISS index from {index_path}...")
index = faiss.read_index(index_path)

# Loading indexed documents
docs_path = os.path.join(INDEX_DIR, "documents.pkl")
with open(docs_path, "rb") as f:
    documents = pickle.load(f)

# Optional: loading a cross-encoder for re-ranking
# Use a re-ranking model like "cross-encoder/ms-marco-MiniLM-L-6-v2"
USE_RERANKING = True
if USE_RERANKING:
    logging.info("Loading cross-encoder for re-ranking...")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def retrieve_documents(query, top_k=DEFAULT_TOP_K, rerank=USE_RERANKING):
    """
    Retrieve the `top_k` closest documents to the query.
    If rerank=True, re-order the results with the cross-encoder.
    """
    query_embedding = embed_model.encode([query])
    distances, indices = index.search(query_embedding, top_k * 3)  # retrieve more for re-ranking
    candidates = [documents[i] for i in indices[0] if i >= 0]
    
    if rerank and candidates:
        logging.info("Re-ranking documents...")
        # Prepare pairs (query, candidate) for the cross-encoder
        pairs = [(query, doc) for doc in candidates]
        scores = reranker.predict(pairs)
        # Sort documents by descending score
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        # Select top_k
        retrieved_texts = [doc for doc, score in ranked[:top_k]]
    else:
        retrieved_texts = candidates[:top_k]
    
    return retrieved_texts

# Command line test
if __name__ == "__main__":
    query = input("🔍 Ask a question: ")
    docs = retrieve_documents(query)
    
    print("\n📚 Relevant documents:")
    for i, doc in enumerate(docs, 1):
        print(f"\n📄 Document {i}:\n{doc[:500]}...\n")
