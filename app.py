#!/usr/bin/env python3
"""
Entry point for the RAG chatbot application.
"""
import argparse
import sys
import logging
import os

def main():
    """Main entry point for the application"""
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Local RAG Chatbot")
    parser.add_argument("--cli", action="store_true", help="Launch the command-line interface")
    parser.add_argument("--web", action="store_true", help="Launch the web interface (recommended)")
    parser.add_argument("--index", action="store_true", help="Index documents and exit")
    args = parser.parse_args()
    
    # Default to web UI if no argument is provided
    if not (args.cli or args.web or args.index):
        args.web = True
    
    # Run the appropriate interface
    if args.index:
        try:
            from src.core.indexing import run_indexing
            data_dir = os.getenv("DATA_DIR", "./data")
            success = run_indexing(data_dir)
            if success:
                logging.info("Indexation completed successfully.")
                logging.info("You can now run the application with --cli, or --web to use the chatbot.")
            else:
                logging.error("Indexation failed.")
                sys.exit(1)
        except Exception as e:
            logging.error(f"Error during indexing: {e}")
            sys.exit(1)
    elif args.cli:
        try:
            # Check if FAISS index exists
            index_dir = os.getenv("INDEX_DIR", "./faiss_index")
            index_path = os.path.join(index_dir, "index.faiss")
            docs_path = os.path.join(index_dir, "documents.pkl")
            
            if not os.path.exists(index_path) or not os.path.exists(docs_path):
                logging.warning("FAISS index not found. Please run 'python app.py --index' first to create an index.")
                logging.warning("Running the CLI interface anyway. The system will prompt you to create an index if needed.")
            
            # Use the refactored CLI module
            from src.ui.cli import run_cli
            run_cli()
        except ImportError as e:
            logging.error(f"Error importing CLI dependencies: {e}")
            sys.exit(1)
        except Exception as e:
            logging.error(f"Error launching the CLI: {e}")
            sys.exit(1)
    else:  # Web UI (default)
        try:            
            # Use the new web UI module
            from src.ui.web_ui import run_web_ui
            logging.info("Starting the web interface...")
            run_web_ui()
        except ImportError as e:
            logging.error(f"Error importing web UI dependencies: {e}")
            logging.error("Please make sure you have installed gradio: pip install gradio>=4.0.0")    
        except Exception as e:
            logging.error(f"Error launching the web UI: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main() 