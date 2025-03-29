#!/usr/bin/env python3
"""
LLM module for the RAG chatbot.
This module handles interactions with language models.
"""
import logging
import os

import ollama
from openai import OpenAI

from src.core.retrieval import retrieve_documents
from src.config.settings import (
    LLM_PROVIDER,
    LLM_MODEL,
    OPENAI_MODEL,
    SYSTEM_PROMPT,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    SUPPORTED_OPENAI_MODELS,
)

# Log configuration
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

class LLMService:
    """Class for handling interactions with language models"""
    
    def __init__(self):
        """Initialize the LLM service with the appropriate client"""
        self.openai_client = None
        
        # Initialize OpenAI client if needed
        if LLM_PROVIDER == "openai":
            self._init_openai_client()
    
    def _init_openai_client(self):
        """Initialize the OpenAI client"""
        api_key = OPENAI_API_KEY
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required when using OpenAI provider")
            
        if OPENAI_MODEL not in SUPPORTED_OPENAI_MODELS:
            logging.warning(f"Model {OPENAI_MODEL} is not in the list of supported models. Using gpt-3.5-turbo as fallback.")
            self.openai_model = "gpt-3.5-turbo"
        else:
            self.openai_model = OPENAI_MODEL
            
        try:
            self.openai_client = OpenAI(
                api_key=api_key,
                base_url=OPENAI_BASE_URL
            )
            logging.info("OpenAI client initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    def ask(self, query):
        """
        Send a query to the LLM with RAG context.
        
        Args:
            query (str): The user query
            
        Returns:
            str: The LLM's response
        """
        try:
            docs = retrieve_documents(query)
            
            # Check if the response contains an error message about missing index
            if len(docs) == 1 and docs[0].startswith("Erreur: L'index RAG n'existe pas"):
                return docs[0] + " Utilisez la commande 'python app.py --index' ou l'interface graphique pour indexer vos documents."
            
            context = "\n\n".join(docs) if docs else "No context found."
        
            prompt = f"{context}\n\nQuestion: {query}"
        
            logging.info(f"Sending prompt to {LLM_PROVIDER}...")
            
            if LLM_PROVIDER == "ollama":
                return self._ollama_query(prompt)
            else:  # OpenAI
                return self._openai_query(prompt)
                
        except Exception as e:
            logging.error(f"Error calling the LLM: {e}")
            return f"Error calling the LLM: {str(e)}"
    
    def _ollama_query(self, prompt):
        """
        Send a query to Ollama.
        
        Args:
            prompt (str): The formatted prompt with context
            
        Returns:
            str: Ollama's response
        """
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
        )
        
        answer = response.get("message", {}).get("content", "No response obtained.")
        logging.info("Response obtained from Ollama.")
        return answer
    
    def _openai_query(self, prompt):
        """
        Send a query to OpenAI.
        
        Args:
            prompt (str): The formatted prompt with context
            
        Returns:
            str: OpenAI's response
        """
        if not self.openai_client:
            self._init_openai_client()
            
        response = self.openai_client.chat.completions.create(
            model=self.openai_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            timeout=30  # 30 seconds timeout
        )
        
        answer = response.choices[0].message.content
        logging.info("Response obtained from OpenAI.")
        return answer

# Create a singleton instance
llm_service = LLMService()

def ask_llm(query):
    """
    Helper function to send a query to the LLM with RAG context.
    
    Args:
        query (str): The user query
        
    Returns:
        str: The LLM's response
    """
    return llm_service.ask(query)

# Command line test
if __name__ == "__main__":
    logging.info(f"Starting the RAG chatbot with {LLM_PROVIDER} ({LLM_MODEL if LLM_PROVIDER == 'ollama' else OPENAI_MODEL})...")
    while True:
        query = input("\n💬 Ask a question (or 'exit' to quit): ").strip()
        if query.lower() == "exit":
            logging.info("Stopping the chatbot.")
            break
        answer = ask_llm(query)
        print("\n🤖 Model's response:")
        print(answer) 