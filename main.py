#!/usr/bin/env python3
import logging
import ollama
from openai import OpenAI
from retrieval import retrieve_documents
from dotenv import load_dotenv
import os

load_dotenv()

# Log configuration
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "mistral")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT")

# List of supported OpenAI models
SUPPORTED_OPENAI_MODELS = [
    "gpt-3.5-turbo",
    "gpt-4",
    "gpt-4-turbo-preview",
    "gpt-4-32k"
]

# Initialize OpenAI client if needed
openai_client = None
if LLM_PROVIDER == "openai":
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required when using OpenAI provider")
    if OPENAI_MODEL not in SUPPORTED_OPENAI_MODELS:
        logging.warning(f"Model {OPENAI_MODEL} is not in the list of supported models. Using gpt-3.5-turbo as fallback.")
        OPENAI_MODEL = "gpt-3.5-turbo"
    try:
        # Using a simpler initialization without additional parameters
        openai_client = OpenAI(
            api_key=api_key,
            base_url="https://api.openai.com/v1"
        )
        logging.info("OpenAI client initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize OpenAI client: {e}")
        raise

def ask_llm(query):
    try:
        docs = retrieve_documents(query)
        context = "\n\n".join(docs) if docs else "No context found."
    
        prompt = f"{context}\n\nQuestion: {query}"
    
        logging.info(f"Sending prompt to {LLM_PROVIDER}...")
        
        if LLM_PROVIDER == "ollama":
            response = ollama.chat(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ]
            )
            answer = response.get("message", {}).get("content", "No response obtained.")
        else:  # OpenAI
            response = openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                timeout=30  # 30 seconds timeout
            )
            answer = response.choices[0].message.content

        logging.info("Response obtained.")
        return answer
    except Exception as e:
        logging.error(f"Error calling the LLM: {e}")
        return f"Error calling the LLM: {str(e)}"

def main():
    logging.info(f"Starting the RAG chatbot with {LLM_PROVIDER} ({LLM_MODEL if LLM_PROVIDER == 'ollama' else OPENAI_MODEL})...")
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
