# 📚 Local RAG Chatbot with Ollama and FAISS

This repository provides a detailed and practical implementation of a Retrieval-Augmented Generation (RAG) chatbot, fully operational on your local machine. It leverages the Ollama ecosystem to interact with powerful Large Language Models (LLMs) such as Mistral 7B or any other model suitable for your hardware resources, and uses FAISS for efficient semantic retrieval. This setup allows you to effortlessly query your personal document collections and obtain accurate, context-aware answers.

---
## How it works

![image](https://github.com/user-attachments/assets/ea7e6be7-24a4-4715-978a-7b1e878eaf6d)


## 🚀 Features

-   **Entirely Local**: Keep your data secure by running everything locally—no cloud required.
-   **Semantic Retrieval with FAISS**: Rapidly retrieve relevant document segments based on semantic similarity.
-   **Customizable LLM Usage**: Choose the LLM best suited to your hardware, such as Mistral 7B, Llama 2, or any other Ollama-supported model.
-   **Supports Multiple Formats**: Easily index `.txt` and `.pdf` files.
-   **Interactive CLI Interface**: User-friendly command-line interaction.

---

## 📖 What is Retrieval-Augmented Generation (RAG)?

Retrieval-Augmented Generation enhances generative AI models by integrating external knowledge dynamically retrieved from a document database during query time. Unlike traditional generative AI, which relies solely on training data, RAG boosts accuracy and contextual relevance by embedding a retrieval step into the generation process.

**Typical Use Cases:**

-   Personalized knowledge management
-   Local documentation search
-   Academic research support
-   Secure and private query handling

---

## ⚙️ Technical Overview

### Workflow

1. **Indexing**: Your documents are segmented into manageable chunks and encoded into numerical vectors using Sentence Transformers.
2. **Semantic Search**: Queries are similarly encoded and matched against document vectors in FAISS.
3. **Re-ranking (optional)**: A cross-encoder further enhances precision by re-ranking retrieved documents.
4. **Response Generation**: The refined context is sent to an Ollama-hosted LLM (e.g., Mistral 7B) to produce an accurate, context-rich response.

---

## 📥 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/CyrilDecostanzi/local_rag_chatbot.git
cd local_rag_chatbot
```

### Step 2: Install Dependencies

Ensure Python 3.8+ is installed.

```bash
pip install -r requirements.txt
```

### Step 3: Install and Set Up Ollama

Follow the official [Ollama installation guide](https://github.com/ollama/ollama).

Pull your desired model (e.g., `mistral`):

```bash
ollama pull mistral
```

---

## ⚙️ Configuration

Create a `.env` file:

```env
# Embedding Model
EMBEDDING_MODEL_NAME="all-MiniLM-L6-v2"

# FAISS Index
INDEX_DIR="faiss_index"
DEFAULT_TOP_K=3

# Re-ranking
USE_RERANKING=true
RERANK_MODEL="cross-encoder/ms-marco-MiniLM-L-6-v2"

# LLM Model
LLM_MODEL="mistral"  # Or your chosen Ollama-supported model

# System Prompt for the LLM
SYSTEM_PROMPT="You are a helpful AI assistant."
```

---

## 📚 Indexing Documents

Place documents in the `./data` folder (default path), then run:

```bash
python index_documents.py
```

This will automatically:

-   Segment documents into semantic chunks.
-   Generate embeddings.
-   Populate the FAISS index.

---

## 🤖 Using the Chatbot

Start your chatbot session:

```bash
python main.py
```

Enter your questions directly into the terminal:

```bash
💬 Ask a question (or 'exit' to quit): How does RAG improve query accuracy?
```

---

## 🛠 Troubleshooting

-   **FAISS Issues**: Ensure compatibility by installing `faiss-cpu` compatible with your OS and Python version.
-   **LLM Issues**: Confirm you've successfully pulled your model via `ollama pull your_model_name`.

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
