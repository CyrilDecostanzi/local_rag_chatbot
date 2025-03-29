#!/usr/bin/env python3
"""
Setup script for Local RAG Chatbot.
This script helps with the installation and configuration of the RAG chatbot system.
"""
import os
import sys
import logging
import subprocess
import shutil
import platform
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def check_python_version():
    """Check if Python version is 3.8 or higher"""
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 8):
        logging.error("Python 3.8 or higher is required.")
        logging.error(f"Current Python version: {sys.version}")
        return False
    return True

def install_requirements():
    """Install requirements from requirements.txt"""
    try:
        logging.info("Installing dependencies from requirements.txt...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error installing dependencies: {e}")
        return False

def create_env_file():
    """Create .env file if it doesn't exist"""
    if os.path.exists(".env"):
        logging.info(".env file already exists. Skipping creation.")
        return True
    
    try:
        logging.info("Creating .env file from template...")
        if os.path.exists(".env.example"):
            shutil.copy(".env.example", ".env")
            logging.info("Created .env file from example template.")
            return True
        else:
            # Create minimal .env file
            with open(".env", "w") as f:
                f.write("""# Embedding Model
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
OPENAI_API_KEY=""
OPENAI_MODEL="gpt-3.5-turbo"

# System Prompt for the LLM
SYSTEM_PROMPT="You are a helpful AI assistant."
""")
            logging.info("Created default .env file.")
            return True
    except Exception as e:
        logging.error(f"Error creating .env file: {e}")
        return False

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = ["data", "faiss_index"]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logging.info(f"Created directory: {directory}")
    
    return True

def check_ollama():
    """Check if Ollama is installed"""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode == 0:
            logging.info("Ollama is installed and running.")
            return True
        else:
            logging.warning("Ollama might be installed but not running.")
            logging.warning("Please run 'ollama serve' in a separate terminal.")
            return False
    except FileNotFoundError:
        logging.warning("Ollama is not installed or not in PATH.")
        logging.warning("Consider installing Ollama: https://github.com/ollama/ollama")
        logging.warning("Or set LLM_PROVIDER=openai in .env to use OpenAI instead.")
        return False
    except Exception as e:
        logging.warning(f"Error checking Ollama: {e}")
        return False

def check_openai_key():
    """Check if OpenAI API key is set"""
    from dotenv import load_dotenv
    load_dotenv()
    
    provider = os.getenv("LLM_PROVIDER", "ollama")
    openai_key = os.getenv("OPENAI_API_KEY", "")
    
    if provider == "openai" and not openai_key:
        logging.warning("LLM_PROVIDER is set to 'openai', but no API key is provided.")
        logging.warning("Please set OPENAI_API_KEY in the .env file.")
        return False
        
    return True

def run_setup():
    """Run the complete setup process"""
    logging.info("Starting setup for Local RAG Chatbot")
    
    if not check_python_version():
        return False
    
    steps = [
        ("Installing dependencies", install_requirements),
        ("Creating configuration file", create_env_file),
        ("Setting up directories", create_directories),
    ]
    
    for step_name, step_func in steps:
        logging.info(f"Step: {step_name}")
        if not step_func():
            logging.error(f"Setup failed at step: {step_name}")
            return False
    
    # Optional checks
    check_ollama()
    check_openai_key()
    
    logging.info("Setup completed successfully!")
    logging.info("\nNext steps:")
    logging.info("1. Add documents to the 'data' directory")
    logging.info("2. Index your documents: python app.py --index")
    logging.info("3. Launch the application: python app.py")
    
    return True

if __name__ == "__main__":
    success = run_setup()
    sys.exit(0 if success else 1) 