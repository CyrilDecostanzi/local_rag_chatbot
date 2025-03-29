#!/usr/bin/env python3
"""
Helper functions for the RAG chatbot.
"""
import os
import logging
import shutil
from pathlib import Path

from src.config.settings import DATA_DIR, INDEX_DIR

def ensure_directory_exists(directory):
    """
    Ensure a directory exists, create if it doesn't.
    
    Args:
        directory (str): Path to the directory
    """
    os.makedirs(directory, exist_ok=True)

def check_index_exists():
    """
    Check if the FAISS index exists.
    
    Returns:
        bool: True if index exists, False otherwise
    """
    index_path = os.path.join(INDEX_DIR, "index.faiss")
    docs_path = os.path.join(INDEX_DIR, "documents.pkl")
    return os.path.exists(index_path) and os.path.exists(docs_path)

def copy_file_to_data_dir(file_path):
    """
    Copy a file to the data directory.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        str: Path to the copied file or None if failed
    """
    try:
        # Ensure data directory exists
        ensure_directory_exists(DATA_DIR)
        
        # Get the file name
        file_name = os.path.basename(file_path)
        
        # Create the destination path
        dest_path = os.path.join(DATA_DIR, file_name)
        
        # Copy the file
        shutil.copy2(file_path, dest_path)
        
        logging.info(f"File {file_name} copied to data directory.")
        return dest_path
    except Exception as e:
        logging.error(f"Error copying file {file_path}: {e}")
        return None

def delete_file_from_data_dir(file_name):
    """
    Delete a file from the data directory.
    
    Args:
        file_name (str): Name of the file to delete
        
    Returns:
        bool: True if file was deleted, False otherwise
    """
    try:
        # Create the file path
        file_path = os.path.join(DATA_DIR, file_name)
        
        # Check if file exists
        if not os.path.exists(file_path):
            logging.error(f"File {file_name} not found in data directory.")
            return False
        
        # Delete the file
        os.remove(file_path)
        
        logging.info(f"File {file_name} deleted from data directory.")
        return True
    except Exception as e:
        logging.error(f"Error deleting file {file_name}: {e}")
        return False

def clear_rag_index():
    """
    Clear the RAG index by removing all files in the index directory.
    
    Returns:
        bool: True if index was cleared, False otherwise
    """
    try:
        if os.path.exists(INDEX_DIR):
            # Remove all files in the index directory
            for file in os.listdir(INDEX_DIR):
                file_path = os.path.join(INDEX_DIR, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            
            logging.info("RAG index cleared successfully.")
            return True
        return False
    except Exception as e:
        logging.error(f"Error clearing RAG index: {e}")
        return False

def list_data_files():
    """
    List all files in the data directory.
    
    Returns:
        list: List of file names in the data directory
    """
    try:
        # Ensure data directory exists
        ensure_directory_exists(DATA_DIR)
        
        # List only files, not directories
        files = [f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))]
        return files
    except Exception as e:
        logging.error(f"Error listing data files: {e}")
        return []

def get_file_info(file_name):
    """
    Get information about a file in the data directory.
    
    Args:
        file_name (str): Name of the file
        
    Returns:
        dict: Dictionary with file information
    """
    try:
        file_path = os.path.join(DATA_DIR, file_name)
        
        if not os.path.exists(file_path):
            return None
        
        # Get file stats
        stats = os.stat(file_path)
        
        # Create info dictionary
        info = {
            "name": file_name,
            "path": file_path,
            "size": stats.st_size,
            "created": stats.st_ctime,
            "modified": stats.st_mtime,
            "type": file_name.split(".")[-1].lower() if "." in file_name else "unknown"
        }
        
        return info
    except Exception as e:
        logging.error(f"Error getting file info for {file_name}: {e}")
        return None 