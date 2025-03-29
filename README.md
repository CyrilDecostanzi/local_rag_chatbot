# 📚 Local RAG Chatbot with FAISS and Ollama (Or cloud-based models via OpenAI)

This repository provides a detailed and practical implementation of a Retrieval-Augmented Generation (RAG) chatbot, fully operational on your local machine. It leverages either the Ollama ecosystem to interact with powerful Large Language Models (LLMs) such as Mistral 7B, or OpenAI's API for cloud-based models. The system uses FAISS for efficient semantic retrieval, allowing you to effortlessly query your personal document collections and obtain accurate, context-aware answers.

---
## How it works

![image](https://github.com/user-attachments/assets/a816b9de-a0dd-406b-b382-0e335ff91902)


## 🚀 Features

-   **Flexible LLM Integration**: Choose between local models via Ollama or cloud-based models via OpenAI
-   **Entirely Local Document Processing**: Keep your document processing and retrieval secure by running everything locally—no cloud required for document handling
-   **Semantic Retrieval with FAISS**: Rapidly retrieve relevant document segments based on semantic similarity
-   **Customizable LLM Usage**: Choose the LLM best suited to your needs:
    -   Local: Mistral 7B, Llama 2, or any other Ollama-supported model
    -   Cloud: GPT-3.5-turbo, GPT-4, or other OpenAI models
-   **Supports Multiple Formats**: Easily index `.txt` and `.pdf` files
-   **Interactive CLI Interface**: User-friendly command-line interaction

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
4. **Response Generation**: The refined context is sent to either:
    - An Ollama-hosted LLM (e.g., Mistral 7B) for local processing
    - OpenAI's API for cloud-based processing

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

### Step 3: Set Up LLM Provider

#### Option A: Local Models with Ollama

Follow the official [Ollama installation guide](https://github.com/ollama/ollama).

Pull your desired model (e.g., `mistral`):

```bash
ollama pull mistral
```

#### Option B: OpenAI API

1. Sign up for an OpenAI account at [OpenAI's website](https://openai.com)
2. Get your API key from the [OpenAI dashboard](https://platform.openai.com/api-keys)

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

# LLM Configuration
# Choose between 'ollama' or 'openai'
LLM_PROVIDER="ollama"

# Ollama settings
LLM_MODEL="mistral"

# OpenAI settings
OPENAI_API_KEY="your-api-key-here"
OPENAI_MODEL="gpt-3.5-turbo"  # Supported models: gpt-3.5-turbo, gpt-4, gpt-4-turbo-preview, gpt-4-32k

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

-   Segment documents into semantic chunks
-   Generate embeddings
-   Populate the FAISS index

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

-   **FAISS Issues**: Ensure compatibility by installing `faiss-cpu` compatible with your OS and Python version
-   **Ollama Issues**: Confirm you've successfully pulled your model via `ollama pull your_model_name`
-   **OpenAI Issues**:
    -   Verify your API key is correctly set in the `.env` file
    -   Check your OpenAI account has sufficient credits
    -   Ensure you're using a supported model

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
