#!/usr/bin/env python3
"""
Command-line interface for the RAG chatbot.
"""
import logging
import os

from src.core.llm import ask_llm
from src.config.settings import LLM_PROVIDER, LLM_MODEL, OPENAI_MODEL, INDEX_DIR

# Log configuration
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

def check_index_exists():
    """
    Check if the FAISS index exists.
    
    Returns:
        bool: True if index exists, False otherwise
    """
    index_path = os.path.join(INDEX_DIR, "index.faiss")
    docs_path = os.path.join(INDEX_DIR, "documents.pkl")
    return os.path.exists(index_path) and os.path.exists(docs_path)

def run_cli():
    """Run the command-line interface for the RAG chatbot"""
    # Check if FAISS index exists
    if not check_index_exists():
        logging.warning("FAISS index not found. Please run 'python app.py --index' first to create an index.")
        logging.warning("Running the CLI interface anyway. The system will prompt you to create an index if needed.")
    
    logging.info(f"Starting the RAG chatbot with {LLM_PROVIDER} ({LLM_MODEL if LLM_PROVIDER == 'ollama' else OPENAI_MODEL})...")
    print("\n🤖 Welcome to the RAG Chatbot! Ask a question or type 'exit' to quit.")
    print("------------------------------------------------------------------")
    
    while True:
        query = input("\n💬 Ask a question (or 'exit' to quit): ").strip()
        if query.lower() == "exit":
            logging.info("Stopping the chatbot.")
            break
        
        answer = ask_llm(query)
        print("\n🤖 Model's response:")
        print(answer)

if __name__ == "__main__":
    run_cli() 