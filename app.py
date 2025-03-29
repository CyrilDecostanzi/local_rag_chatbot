#!/usr/bin/env python3
import argparse
import sys
import logging

def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Local RAG Chatbot")
    parser.add_argument("--cli", action="store_true", help="Launch the command-line interface")
    parser.add_argument("--gui", action="store_true", help="Launch the graphical user interface (Tkinter)")
    parser.add_argument("--index", action="store_true", help="Index documents and exit")
    args = parser.parse_args()
    
    # Default to GUI if no argument is provided
    if not (args.cli or args.gui or args.index):
        args.gui = True
    
    # Import modules based on the selected interface
    if args.index:
        from index_documents import main as index_main
        # Call with default arguments
        index_main(argparse.Namespace(data_path="./data"))
    elif args.cli:
        from main import main as cli_main
        cli_main()
    else:  # GUI
        try:
            import tkinter as tk
            from tkinter_app import RAGChatbotApp
            
            # Start the Tkinter application
            root = tk.Tk()
            app = RAGChatbotApp(root)
            root.mainloop()
        except ImportError as e:
            logging.error(f"Error importing Tkinter dependencies: {e}")
            logging.error("Please make sure you have installed Tkinter and Pillow.")
            sys.exit(1)
        except Exception as e:
            logging.error(f"Error launching the GUI: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main() 