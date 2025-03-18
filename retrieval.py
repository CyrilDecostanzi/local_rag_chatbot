#!/usr/bin/env python3
import os
import pickle
import logging
import operator

import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv
import os

load_dotenv()

# Log configuration
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

# Parameters
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
INDEX_DIR = os.getenv("INDEX_DIR", "faiss_index")
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", 3))
RETRIEVAL_MULTIPLIER = int(os.getenv("RETRIEVAL_MULTIPLIER", 3))
USE_RERANKING = os.getenv("USE_RERANKING", "True").lower() == "true"
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

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
if USE_RERANKING:
    logging.info("Loading cross-encoder for re-ranking...")
    reranker = CrossEncoder(RERANK_MODEL)

def retrieve_documents(query, top_k=DEFAULT_TOP_K, rerank=USE_RERANKING):  
    # Retrieve the `top_k` closest documents to the query.
    # If rerank=True, re-order the results with the cross-encoder.
    query_embedding = embed_model.encode([query])
    D, I = index.search(query_embedding, top_k * RETRIEVAL_MULTIPLIER)
    candidates = [documents[i] for i in I[0] if 0 <= i < len(documents)]
    
    if rerank and candidates:
        logging.info("Re-ranking documents...")
        pairs = [(query, doc) for doc in candidates]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=operator.itemgetter(1), reverse=True)
        batch_size = 32
        scores = []
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            batch_scores = reranker.predict(batch_pairs)
            scores.extend(batch_scores)
            batch_scores = reranker.predict(batch_pairs)
            scores.extend(batch_scores)
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
