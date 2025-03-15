#!/usr/bin/env python3
import logging
import ollama
from retrieval import retrieve_documents

# Log configuration
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

SYSTEM_PROMPT = """Tu es un assistant IA, tu dois fournir des informations précises sur les documents qu'on te fourni, tu es spécialiste automobile et tu dois citer les source dans les documents que tu consulte. Tu repond en français, si l'information n'est pas présente tu ne l'invente pas repond simplement que tu ne dispose pas de l'information."""


def ask_llm(query):
    """
    Constructs a prompt enriched with retrieved documents and queries the LLM.
    """
    try:
        docs = retrieve_documents(query)
        context = "\n\n".join(docs) if docs else "No context found."
    
        prompt = f"""
You are an intelligent assistant. Here is some relevant information:
    
{context}

Now, answer this question in detail:
{query}
        """.strip()
    
        logging.info("Sending prompt to the LLM...")
        response = ollama.chat(model="mistral-small", messages=[{"role": "user", "content": prompt},{"role": "system", "content": SYSTEM_PROMPT},])
        return response.get("message", {}).get("content", "No response obtained.")
    except Exception as e:
        logging.error(f"Error calling the LLM: {e}")
        return "Error calling the LLM."

def main():
    logging.info("Starting the RAG chatbot with Mistral 7B...")
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
