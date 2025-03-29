#!/usr/bin/env python3
"""
Main entry point for the RAG chatbot application.
"""
import argparse
import sys
import logging
import os

from src.config.settings import INDEX_DIR
from src.utils.helpers import check_index_exists
from src.core.indexing import run_indexing

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

def main():
    """Main entry point for the application"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Local RAG Chatbot")
    parser.add_argument("--cli", action="store_true", help="Launch the command-line interface")
    parser.add_argument("--gui", action="store_true", help="Launch the graphical user interface (Tkinter)")
    parser.add_argument("--index", action="store_true", help="Index documents and exit")
    args = parser.parse_args()
    
    # Default to GUI if no argument is provided
    if not (args.cli or args.gui or args.index):
        args.gui = True
    
    # Run the appropriate interface
    if args.index:
        try:
            from src.config.settings import DATA_DIR
            # Call indexing function
            success = run_indexing(DATA_DIR)
            if success:
                logging.info("Indexation completed successfully.")
                logging.info("You can now run the application with --cli or --gui to use the chatbot.")
            else:
                logging.error("Indexation failed.")
                sys.exit(1)
        except Exception as e:
            logging.error(f"Error during indexing: {e}")
            sys.exit(1)
    elif args.cli:
        try:
            # Check if FAISS index exists
            if not check_index_exists():
                logging.warning("FAISS index not found. Please run 'python main.py --index' first to create an index.")
                logging.warning("Running the CLI interface anyway. The system will prompt you to create an index if needed.")
            
            from src.ui.cli import run_cli
            run_cli()
        except ImportError as e:
            logging.error(f"Error importing CLI dependencies: {e}")
            sys.exit(1)
        except Exception as e:
            logging.error(f"Error launching the CLI: {e}")
            sys.exit(1)
    else:  # GUI
        try:
            import tkinter as tk
            from src.ui.gui import run_gui
            
            # Start the Tkinter application
            run_gui()
        except ImportError as e:
            logging.error(f"Error importing Tkinter dependencies: {e}")
            logging.error("Please make sure you have installed Tkinter and Pillow.")
            sys.exit(1)
        except Exception as e:
            logging.error(f"Error launching the GUI: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main() 