#!/usr/bin/env python3
"""
Settings module for the RAG chatbot.
This module centralizes all application settings and loads from environment variables.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
INDEX_DIR = os.getenv("INDEX_DIR", os.path.join(BASE_DIR, "faiss_index"))

# LLM settings
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "mistral")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are a helpful AI assistant.")

# OpenAI specific settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# Embedding model settings
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# Document chunking settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# Retrieval settings
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "3"))
RETRIEVAL_MULTIPLIER = int(os.getenv("RETRIEVAL_MULTIPLIER", "3"))
USE_RERANKING = os.getenv("USE_RERANKING", "True").lower() == "true"
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Supported OpenAI models
SUPPORTED_OPENAI_MODELS = [
    "gpt-3.5-turbo",
    "gpt-4",
    "gpt-4-turbo-preview",
    "gpt-4-32k"
]

# UI Color scheme
COLORS = {
    "primary": "#1e3a8a",       # Dark blue for headers
    "primary_light": "#3b82f6",  # Light blue for selected elements
    "secondary": "#0284c7",     # Teal blue for buttons and accents
    "secondary_hover": "#0369a1", # Darker teal for button hover
    "accent": "#ef4444",        # Red for alerts or important elements
    "success": "#10b981",       # Green for success messages
    "warning": "#f59e0b",       # Amber for warnings
    "background": "#f8fafc",    # Very light gray for backgrounds
    "surface": "#ffffff",       # White for card backgrounds
    "text": "#0f172a",          # Very dark blue for text
    "text_secondary": "#475569", # Medium gray for secondary text
    "text_light": "#94a3b8",    # Light gray for tertiary text
    "border": "#e2e8f0",        # Light gray for borders
    "user_message": "#dbeafe",  # Light blue for user messages
    "bot_message": "#f1f5f9",   # Light gray for bot messages
    "disabled": "#cbd5e1",      # Gray for disabled elements
}

# Font settings
FONTS = {
    "title": ("Helvetica", 18, "bold"),
    "subtitle": ("Helvetica", 14, "bold"),
    "body": ("Helvetica", 10),
    "body_bold": ("Helvetica", 10, "bold"),
    "small": ("Helvetica", 9),
    "tiny": ("Helvetica", 8),
    "button": ("Helvetica", 10, "bold"),
}

# UI dimensions
DIMENSIONS = {
    "padding_small": 5,
    "padding": 10,
    "padding_large": 20,
    "border_radius": 8,
    "input_height": 3,
    "button_width": 120,
} 