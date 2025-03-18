#!/usr/bin/env python3
import logging
import ollama
from retrieval import retrieve_documents
from dotenv import load_dotenv
import os

load_dotenv()


# Log configuration
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

LLM_MODEL = os.getenv("LLM_MODEL", "mistral")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT")

def ask_llm(query):
    try:
        docs = retrieve_documents(query)
        context = "\n\n".join(docs) if docs else "No context found."
    
        prompt = f"{context}\n\nQuestion: {query}"
    
        logging.info("Sending prompt to the LLM...")
        response = ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt},{"role": "system", "content": SYSTEM_PROMPT},])
        logging.info("Response obtained.")
        logging.info(response)
        return response.get("message", {}).get("content", "No response obtained.")
    except Exception as e:
        logging.error(f"Error calling the LLM: {e}")
        return "Error calling the LLM."

def main():
    logging.info(f"Starting the RAG chatbot with {LLM_MODEL}...")
    while True:
        query = input("\n💬 Ask a question (or 'exit' to quit): ").strip()
        if query.lower() == "exit":
            logging.info("Stopping the chatbot.")
            break
        answer = ask_llm(query)
        print("\n🤖 Model's response:")
        print(answer)

if __name__ == "__main__":
    main()
