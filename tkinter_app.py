#!/usr/bin/env python3
import os
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import tkinter.font as tkFont
from PIL import Image, ImageTk
import threading
import logging
import subprocess
import shutil
from pathlib import Path
import time

# Imports from the existing project
from retrieval import retrieve_documents
from dotenv import load_dotenv
import ollama
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "mistral")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are a helpful AI assistant.")

# FAISS and document configuration
DATA_DIR = os.getenv("DATA_DIR", "./data")
INDEX_DIR = os.getenv("INDEX_DIR", "faiss_index")

# Color scheme
COLORS = {
    "primary": "#1e3a8a",       # Dark blue for headers - plus profond et élégant
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

# Initialize OpenAI client if needed
openai_client = None
if LLM_PROVIDER == "openai":
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required when using OpenAI provider")
    try:
        openai_client = OpenAI(
            api_key=api_key,
            base_url="https://api.openai.com/v1"
        )
        logging.info("OpenAI client initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize OpenAI client: {e}")
        raise

class ScrollableFrame(ttk.Frame):
    """A scrollable frame widget"""
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        
        # Create a canvas and scrollbar
        self.canvas = tk.Canvas(self, background=COLORS["background"], highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        
        # Configure the canvas
        self.scrollable_frame = ttk.Frame(self.canvas)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        # Create window inside canvas
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Expand canvas on resize
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        
        # Add mouse wheel bindings for scrolling
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel_windows)  # Windows
        self.canvas.bind_all("<Button-4>", self._on_mousewheel_linux_up)  # Linux scroll up
        self.canvas.bind_all("<Button-5>", self._on_mousewheel_linux_down)  # Linux scroll down
        
        # Pack everything
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
    
    def on_canvas_configure(self, event):
        """Resize the canvas window when the canvas is resized"""
        self.canvas.itemconfig(self.canvas_window, width=event.width)
    
    def _on_mousewheel_windows(self, event):
        """Handle mousewheel event on Windows"""
        if self.winfo_containing(event.x_root, event.y_root) == self.canvas:
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def _on_mousewheel_linux_up(self, event):
        """Handle mousewheel up event on Linux"""
        if self.winfo_containing(event.x_root, event.y_root) == self.canvas:
            self.canvas.yview_scroll(-1, "units")
    
    def _on_mousewheel_linux_down(self, event):
        """Handle mousewheel down event on Linux"""
        if self.winfo_containing(event.x_root, event.y_root) == self.canvas:
            self.canvas.yview_scroll(1, "units")

class MessageBubble(ttk.Frame):
    """Custom message bubble widget for chat interface"""
    def __init__(self, parent, message, is_user=False, **kwargs):
        super().__init__(parent, **kwargs)
        
        # Set background color based on message type
        bg_color = COLORS["user_message"] if is_user else COLORS["bot_message"]
        
        # Create a container frame
        container = ttk.Frame(self)
        container.pack(fill="x", expand=True)
        
        # Create a timestamp (small and subtle)
        timestamp = time.strftime("%H:%M")
        
        # Person indicator
        if is_user:
            sender_label = tk.Label(
                container,
                text="Vous",
                font=("Helvetica", 8, "bold"),
                fg=COLORS["text_light"],
                bg=COLORS["background"],
                anchor="e"
            )
            sender_label.pack(side="right", padx=(0, 10), pady=(0, 2))
        else:
            sender_label = tk.Label(
                container,
                text="Assistant",
                font=("Helvetica", 8, "bold"),
                fg=COLORS["text_light"],
                bg=COLORS["background"],
                anchor="w"
            )
            sender_label.pack(side="left", padx=(10, 0), pady=(0, 2))
        
        # Create the bubble frame with rounded corners effect
        bubble_frame = tk.Frame(
            self, 
            bg=bg_color,
            highlightbackground=COLORS["secondary"] if is_user else COLORS["primary"],
            highlightthickness=1,
            bd=0
        )
        
        # Position the bubble left or right based on sender
        bubble_frame.pack(
            side="right" if is_user else "left",
            anchor="e" if is_user else "w",
            padx=10,
            pady=2,
            fill="x",
            expand=False
        )
        
        # Create a label to hold the message
        msg_label = tk.Label(
            bubble_frame, 
            text=message,
            justify=tk.LEFT,
            wraplength=400,
            bg=bg_color,
            fg=COLORS["text"],
            padx=12,
            pady=10,
            font=("Helvetica", 10)
        )
        msg_label.pack(fill="both", expand=True)
        
        # Time indicator (small and subtle)
        time_label = tk.Label(
            bubble_frame,
            text=timestamp,
            font=("Helvetica", 7),
            fg=COLORS["text_light"],
            bg=bg_color,
            anchor="e" if is_user else "w"
        )
        time_label.pack(side="bottom", anchor="e" if is_user else "w", padx=5, pady=(0, 2))

class RAGChatbotApp:
    """Main application class for the RAG chatbot GUI"""
    def __init__(self, root):
        self.root = root
        self.root.title("Local RAG Chatbot")
        self.root.geometry("1000x700")
        self.root.configure(bg=COLORS["background"])
        
        # Set window icon if available
        try:
            self.root.iconbitmap("icon.ico")
        except:
            pass  # No icon file available
        
        # Add window minimum size
        self.root.minsize(800, 600)
        
        # Set custom font
        self.default_font = tkFont.nametofont("TkDefaultFont")
        self.default_font.configure(family="Helvetica", size=10)
        self.root.option_add("*Font", self.default_font)
        
        # Initialize chat history
        self.chat_history = []
        
        # Set up styles
        self.setup_styles()
        
        # Create main frame
        self.main_frame = ttk.Frame(root, style="Main.TFrame")
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create sidebar and content frames
        self.setup_layout()
        
        # Configure the chat interface
        self.setup_chat_interface()
        
        # Configure the file upload area
        self.setup_file_upload()
        
        # Configure the settings panel
        self.setup_settings_panel()
        
        # Setup keyboard shortcuts
        self.setup_shortcuts()
        
        # Check if index exists
        self.check_index()
        
        # Add a small fade-in effect
        self.root.attributes("-alpha", 0.0)
        self.fade_in()
    
    def setup_styles(self):
        """Configure TTK styles for the application"""
        self.style = ttk.Style()
        
        # Frame styles
        self.style.configure("TFrame", background=COLORS["background"])
        self.style.configure("Main.TFrame", background=COLORS["background"])
        self.style.configure("Sidebar.TFrame", background=COLORS["primary"])
        self.style.configure("Content.TFrame", background=COLORS["background"])
        self.style.configure("Card.TFrame", background=COLORS["surface"], 
                            relief="flat", borderwidth=1)
        
        # Button styles - modern and flat
        self.style.configure("TButton", 
                            background=COLORS["secondary"],
                            foreground="white", 
                            font=FONTS["button"],
                            borderwidth=0,
                            focusthickness=0,
                            padding=DIMENSIONS["padding"])
        
        self.style.map("TButton",
                    background=[('active', COLORS["secondary_hover"]), 
                               ('pressed', COLORS["secondary_hover"]), 
                               ('disabled', COLORS["disabled"])],
                    foreground=[('disabled', COLORS["text_light"])])
        
        # Accent button (e.g. for important actions)
        self.style.configure("Accent.TButton",
                            background=COLORS["accent"],
                            foreground="white")
        
        self.style.map("Accent.TButton",
                     background=[('active', "#c81e1e"), 
                                ('pressed', "#c81e1e")])
        
        # Success button
        self.style.configure("Success.TButton",
                            background=COLORS["success"],
                            foreground="white")
        
        self.style.map("Success.TButton",
                     background=[('active', "#059669"), 
                                ('pressed', "#059669")])
        
        # Label styles
        self.style.configure("TLabel", background=COLORS["background"], 
                           foreground=COLORS["text"], font=FONTS["body"])
        self.style.configure("Title.TLabel", font=FONTS["title"], 
                            background=COLORS["background"], foreground=COLORS["primary"])
        self.style.configure("Subtitle.TLabel", font=FONTS["subtitle"], 
                             background=COLORS["background"], foreground=COLORS["text"])
        self.style.configure("Sidebar.TLabel", background=COLORS["primary"], 
                            foreground="white", font=FONTS["body"])
        self.style.configure("SidebarTitle.TLabel", background=COLORS["primary"], 
                            foreground="white", font=FONTS["subtitle"])
        
        # Entry and text input styles
        self.style.configure("TEntry", 
                            fieldbackground=COLORS["surface"],
                            background=COLORS["surface"],
                            foreground=COLORS["text"],
                            bordercolor=COLORS["border"],
                            lightcolor=COLORS["border"],
                            darkcolor=COLORS["border"],
                            insertcolor=COLORS["text"],
                            borderwidth=1)
        
        # Combobox styles
        self.style.configure("TCombobox", 
                            background=COLORS["surface"],
                            foreground=COLORS["text"],
                            fieldbackground=COLORS["surface"],
                            selectbackground=COLORS["primary_light"],
                            selectforeground="white")
        
        # Tab styles
        self.style.configure("TNotebook", background=COLORS["background"], borderwidth=0)
        self.style.configure("TNotebook.Tab", 
                            background=COLORS["primary"], 
                            foreground="white", 
                            padding=[15, 7],
                            font=FONTS["body_bold"])
        
        self.style.map("TNotebook.Tab",
                     background=[("selected", COLORS["primary_light"])],
                     foreground=[("selected", "white")],
                     expand=[("selected", [1, 1, 1, 0])])
        
        # Separator style
        self.style.configure("TSeparator", background=COLORS["border"])
        
        # Custom styles for various elements
        self.style.configure("Link.TLabel", 
                            foreground=COLORS["primary_light"], 
                            font=FONTS["body_bold"],
                            background=COLORS["background"])
        
        self.style.configure("Card.TFrame", background=COLORS["surface"], 
                            relief="flat", borderwidth=1, padding=DIMENSIONS["padding"])
    
    def setup_layout(self):
        """Create the main layout with sidebar and content area"""
        # Create sidebar
        self.sidebar = ttk.Frame(self.main_frame, style="Sidebar.TFrame", width=200)
        self.sidebar.pack(side="left", fill="y", padx=0, pady=0)
        
        # App title in sidebar
        title_frame = ttk.Frame(self.sidebar, style="Sidebar.TFrame")
        title_frame.pack(fill="x", padx=10, pady=20)
        
        app_title = ttk.Label(title_frame, text="RAG Chatbot", style="Sidebar.TLabel", font=("Helvetica", 16, "bold"))
        app_title.pack(anchor="w")
        
        app_subtitle = ttk.Label(title_frame, text="Local Knowledge Base", style="Sidebar.TLabel")
        app_subtitle.pack(anchor="w")
        
        # Sidebar navigation buttons
        self.nav_buttons_frame = ttk.Frame(self.sidebar, style="Sidebar.TFrame")
        self.nav_buttons_frame.pack(fill="x", padx=10, pady=10)
        
        # Define navigation button names and corresponding functions
        nav_items = [
            ("Chat", self.show_chat_tab),
            ("Documents", self.show_documents_tab),
            ("Settings", self.show_settings_tab)
        ]
        
        # Create navigation buttons
        self.nav_buttons = {}
        for text, command in nav_items:
            btn = ttk.Button(self.nav_buttons_frame, text=text, command=command)
            btn.pack(fill="x", pady=5)
            self.nav_buttons[text] = btn
        
        # Create content area with notebook (tabbed interface)
        self.content = ttk.Frame(self.main_frame, style="Content.TFrame")
        self.content.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        # Create notebook for tabbed interface
        self.notebook = ttk.Notebook(self.content)
        self.notebook.pack(fill="both", expand=True)
        
        # Create tabs
        self.chat_tab = ttk.Frame(self.notebook, style="TFrame")
        self.documents_tab = ttk.Frame(self.notebook, style="TFrame")
        self.settings_tab = ttk.Frame(self.notebook, style="TFrame")
        
        self.notebook.add(self.chat_tab, text="Chat")
        self.notebook.add(self.documents_tab, text="Documents")
        self.notebook.add(self.settings_tab, text="Settings")
        
        # Hide the actual tabs, we'll use the sidebar buttons instead
        self.notebook.hide(0)
        self.notebook.hide(1)
        self.notebook.hide(2)
    
    def setup_chat_interface(self):
        """Set up the chat interface tab"""
        # Create main container for chat area
        chat_container = ttk.Frame(self.chat_tab, style="TFrame")
        chat_container.pack(fill="both", expand=True, padx=DIMENSIONS["padding"], pady=DIMENSIONS["padding"])
        
        # Create a header with title and info
        header_frame = ttk.Frame(chat_container, style="TFrame")
        header_frame.pack(fill="x", pady=(0, DIMENSIONS["padding"]))
        
        chat_title = ttk.Label(
            header_frame,
            text="Chat avec documents",
            style="Title.TLabel"
        )
        chat_title.pack(side="left", anchor="w")
        
        # Add a status indicator that shows if index exists
        self.chat_status_frame = ttk.Frame(header_frame, style="TFrame")
        self.chat_status_frame.pack(side="right", anchor="e", padx=DIMENSIONS["padding"])
        
        self.status_indicator = tk.Canvas(
            self.chat_status_frame, 
            width=12, 
            height=12, 
            bg=COLORS["background"], 
            highlightthickness=0
        )
        self.status_indicator.pack(side="left", padx=(0, 5))
        
        self.status_label = ttk.Label(
            self.chat_status_frame,
            text="Vérification de l'index...",
            style="TLabel"
        )
        self.status_label.pack(side="left")
        
        # Add a separator
        ttk.Separator(chat_container, orient="horizontal").pack(fill="x", pady=DIMENSIONS["padding_small"])
        
        # Create a card-like container for the chat
        chat_card = ttk.Frame(chat_container, style="Card.TFrame")
        chat_card.pack(fill="both", expand=True, pady=DIMENSIONS["padding_small"])
        
        # Create scrollable message area
        messages_container = ttk.Frame(chat_card, style="TFrame")
        messages_container.pack(fill="both", expand=True, padx=DIMENSIONS["padding"], 
                               pady=(DIMENSIONS["padding"], 0))
        
        self.messages_frame = ScrollableFrame(messages_container)
        self.messages_frame.pack(fill="both", expand=True)
        
        # Add a separator between messages and input
        ttk.Separator(chat_card, orient="horizontal").pack(fill="x", padx=DIMENSIONS["padding"])
        
        # Create frame for input and send button with improved styling
        input_frame = ttk.Frame(chat_card, style="TFrame", padding=DIMENSIONS["padding"])
        input_frame.pack(fill="x", side="bottom")
        
        # Create text input with better styling
        self.message_input = scrolledtext.ScrolledText(
            input_frame, 
            height=DIMENSIONS["input_height"], 
            font=FONTS["body"],
            wrap=tk.WORD,
            bg=COLORS["surface"],
            fg=COLORS["text"],
            insertbackground=COLORS["text"],  # cursor color
            borderwidth=1,
            relief="solid",
            highlightcolor=COLORS["primary_light"],
            highlightbackground=COLORS["border"],
            highlightthickness=1
        )
        self.message_input.pack(side="left", fill="both", expand=True, padx=(0, DIMENSIONS["padding"]))
        
        # Set placeholder text
        self.message_input.insert("1.0", "Posez votre question ici...")
        self.message_input.configure(fg=COLORS["text_light"])
        
        # Handle focus events for placeholder text
        self.message_input.bind("<FocusIn>", self.on_entry_focus_in)
        self.message_input.bind("<FocusOut>", self.on_entry_focus_out)
        self.message_input.bind("<Return>", self.on_send_press)
        self.message_input.bind("<Shift-Return>", lambda e: "break")  # Allow newlines with Shift+Enter
        
        # Create send button with icon-like appearance
        send_button_frame = ttk.Frame(input_frame, style="TFrame")
        send_button_frame.pack(side="right", padx=0)
        
        self.send_button = ttk.Button(
            send_button_frame,
            text="Envoyer",
            command=self.send_message,
            style="TButton",
            width=10
        )
        self.send_button.pack(side="right")
        
        # Welcome message will be added after index check
    
    def on_entry_focus_in(self, event):
        """Handle focus in event for input field, clear placeholder"""
        if self.message_input.get("1.0", "end-1c").strip() == "Posez votre question ici...":
            self.message_input.delete("1.0", tk.END)
            self.message_input.configure(fg=COLORS["text"])
    
    def on_entry_focus_out(self, event):
        """Handle focus out event for input field, restore placeholder if empty"""
        if not self.message_input.get("1.0", "end-1c").strip():
            self.message_input.delete("1.0", tk.END)
            self.message_input.insert("1.0", "Posez votre question ici...")
            self.message_input.configure(fg=COLORS["text_light"])
    
    def check_index(self):
        """Check if FAISS index exists and update UI accordingly"""
        index_path = os.path.join(INDEX_DIR, "index.faiss")
        docs_path = os.path.join(INDEX_DIR, "documents.pkl")
        
        # Draw status indicator
        if hasattr(self, 'status_indicator'):
            self.status_indicator.delete("all")  # Clear previous drawing
        
        if os.path.exists(index_path) and os.path.exists(docs_path):
            # Update status indicator and label if they exist
            if hasattr(self, 'status_indicator'):
                # Draw green circle
                self.status_indicator.create_oval(2, 2, 10, 10, fill=COLORS["success"], outline="")
                self.status_label.configure(text="Index prêt", foreground=COLORS["success"])
            
            # Add welcome message in chat
            self.add_message("✅ Index FAISS détecté. Prêt à répondre à vos questions sur les documents.", False)
        else:
            # Update status indicator and label if they exist
            if hasattr(self, 'status_indicator'):
                # Draw amber circle
                self.status_indicator.create_oval(2, 2, 10, 10, fill=COLORS["warning"], outline="")
                self.status_label.configure(text="Index non trouvé", foreground=COLORS["warning"])
            
            # Add warning message in chat
            self.add_message("⚠️ Aucun index FAISS trouvé. Veuillez ajouter des documents et les indexer dans l'onglet Documents.", False)
    
    def on_send_press(self, event):
        """Handle Enter key press in the input field"""
        if not event.state & 0x1:  # If Shift key is not pressed
            self.send_message()
            return "break"  # Prevents the newline from being added
        return None
    
    def add_message(self, message, is_user=True):
        """Add a message to the chat interface"""
        message_bubble = MessageBubble(self.messages_frame.scrollable_frame, message, is_user)
        message_bubble.pack(fill="x", pady=5)
        
        # Auto-scroll to the new message
        self.messages_frame.canvas.update_idletasks()
        self.messages_frame.canvas.yview_moveto(1.0)
        
        # Store in chat history
        self.chat_history.append({"role": "user" if is_user else "assistant", "content": message})
    
    def send_message(self):
        """Send a message and get a response from the LLM"""
        message = self.message_input.get("1.0", "end-1c").strip()
        if message == "Posez votre question ici...":
            message = ""
        
        if not message:
            return
        
        # Add user message to chat
        self.add_message(message, True)
        
        # Clear input field
        self.message_input.delete("1.0", tk.END)
        self.message_input.configure(fg=COLORS["text"])
        
        # Create a typing indicator with modern styling
        typing_frame = ttk.Frame(self.messages_frame.scrollable_frame)
        typing_frame.pack(fill="x", pady=5, padx=10, anchor="w")
        
        typing_indicator = ttk.Frame(typing_frame, style="Card.TFrame")
        typing_indicator.pack(side="left", padx=5, pady=5)
        
        # Create dots animation effect
        dots_label = tk.Label(
            typing_indicator,
            text="L'IA réfléchit...",
            bg=COLORS["surface"],
            fg=COLORS["text_secondary"],
            padx=10,
            pady=8,
            font=FONTS["body"]
        )
        dots_label.pack(side="left")
        
        # Animation data
        self.dot_count = 0
        
        def animate_dots():
            if not typing_frame.winfo_exists():
                return
            
            try:
                dots = "." * ((self.dot_count % 3) + 1)
                dots_label.configure(text=f"L'IA réfléchit{dots}")
                self.dot_count += 1
                self.root.after(500, animate_dots)
            except tk.TclError:
                # Widget was destroyed
                pass
        
        # Start animation
        animate_dots()
        
        # Update UI before making LLM call
        self.root.update()
        
        # Get response in a separate thread
        threading.Thread(target=self.get_llm_response, args=(message, typing_frame), daemon=True).start()
    
    def get_llm_response(self, query, typing_frame):
        """Get response from the LLM in a separate thread"""
        try:
            # Check if FAISS index exists
            index_path = os.path.join(INDEX_DIR, "index.faiss")
            if not os.path.exists(index_path):
                response = "⚠️ Aucun index FAISS trouvé. Veuillez ajouter des documents et les indexer dans l'onglet Documents."
            else:
                try:
                    # Retrieve documents
                    docs = retrieve_documents(query)
                    context = "\n\n".join(docs) if docs else "Aucun contexte pertinent trouvé."
                    
                    # Create prompt
                    prompt = f"{context}\n\nQuestion: {query}"
                    
                    # Get provider and model
                    provider = self.llm_provider_var.get()
                    model = self.model_var.get()
                    system_prompt = self.system_prompt_text.get("1.0", "end-1c").strip()
                    
                    # Get response from LLM
                    if provider == "ollama":
                        response_obj = ollama.chat(
                            model=model,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": prompt}
                            ]
                        )
                        response = response_obj.get("message", {}).get("content", "Aucune réponse obtenue.")
                    else:  # OpenAI
                        # Reinitialize client with current API key
                        api_key = self.api_key_var.get()
                        if not api_key:
                            response = "Erreur: La clé API OpenAI n'est pas définie. Veuillez la configurer dans les Paramètres."
                        else:
                            client = OpenAI(
                                api_key=api_key,
                                base_url="https://api.openai.com/v1"
                            )
                            response_obj = client.chat.completions.create(
                                model=model,
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": prompt}
                                ]
                            )
                            response = response_obj.choices[0].message.content
                
                except Exception as e:
                    logging.error(f"Error calling the LLM: {e}")
                    response = f"Erreur: {str(e)}"
        
        except Exception as e:
            logging.error(f"Error processing query: {e}")
            response = f"Erreur: {str(e)}"
        
        # Update UI in the main thread
        self.root.after(0, lambda: self.update_chat_with_response(response, typing_frame))
    
    def update_chat_with_response(self, response, typing_frame):
        """Update the chat with the LLM response"""
        # Remove typing indicator
        if typing_frame.winfo_exists():
            typing_frame.destroy()
        
        # Add bot response
        self.add_message(response, False)
        
        # Play sound effect for completed response (optional)
        self.root.bell()
    
    def upload_document(self):
        """Upload a document to the data directory"""
        file_types = [
            ("Documents texte", "*.txt"),
            ("Documents PDF", "*.pdf"),
            ("Tous les fichiers", "*.*")
        ]
        
        file_paths = filedialog.askopenfilenames(
            title="Sélectionnez les documents à importer",
            filetypes=file_types
        )
        
        if not file_paths:
            return
        
        # Create progress window
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Importation des documents")
        progress_window.geometry("400x150")
        progress_window.transient(self.root)
        progress_window.grab_set()
        progress_window.configure(bg=COLORS["background"])
        
        # Center the window
        progress_window.update_idletasks()
        width = progress_window.winfo_width()
        height = progress_window.winfo_height()
        x = (self.root.winfo_width() // 2) - (width // 2) + self.root.winfo_x()
        y = (self.root.winfo_height() // 2) - (height // 2) + self.root.winfo_y()
        progress_window.geometry(f"+{x}+{y}")
        
        # Label
        info_label = tk.Label(
            progress_window, 
            text=f"Importation de {len(file_paths)} fichier(s)...", 
            bg=COLORS["background"],
            fg=COLORS["text"],
            font=FONTS["body_bold"],
            pady=10
        )
        info_label.pack(pady=10)
        
        # Progress bar
        progress = ttk.Progressbar(
            progress_window, 
            orient="horizontal", 
            length=350, 
            mode="determinate"
        )
        progress.pack(pady=10, padx=20)
        
        # Status label
        status_label = tk.Label(
            progress_window, 
            text="Démarrage de l'importation...", 
            bg=COLORS["background"],
            fg=COLORS["text_secondary"],
            font=FONTS["small"]
        )
        status_label.pack(pady=5)
        
        # Ensure data directory exists
        os.makedirs(DATA_DIR, exist_ok=True)
        
        def copy_files():
            # Set maximum value for progress bar
            progress["maximum"] = len(file_paths)
            
            # Copy files to data directory
            for i, file_path in enumerate(file_paths):
                if not progress_window.winfo_exists():
                    return  # Window was closed
                    
                file_name = os.path.basename(file_path)
                destination = os.path.join(DATA_DIR, file_name)
                
                status_label.configure(text=f"Importation de {file_name}...")
                
                try:
                    shutil.copy2(file_path, destination)
                    progress["value"] = i + 1
                    progress_window.update()
                    logging.info(f"Copied {file_path} to {destination}")
                except Exception as e:
                    logging.error(f"Error copying file {file_path}: {e}")
                    tk.messagebox.showerror(
                        "Erreur d'importation",
                        f"Erreur lors de l'importation de {file_name}: {str(e)}"
                    )
            
            # Close progress window
            if progress_window.winfo_exists():
                progress_window.destroy()
                
                # Show success message
                tk.messagebox.showinfo(
                    "Importation terminée",
                "Error",
                f"Error saving settings: {str(e)}"
            )
    
    def show_chat_tab(self):
        """Show the chat tab"""
        self.notebook.select(0)
    
    def show_documents_tab(self):
        """Show the documents tab"""
        self.notebook.select(1)
    
    def show_settings_tab(self):
        """Show the settings tab"""
        self.notebook.select(2)

    def setup_file_upload(self):
        """Set up the documents tab for file uploads"""
        # Create main container
        doc_container = ttk.Frame(self.documents_tab, style="TFrame")
        doc_container.pack(fill="both", expand=True, padx=DIMENSIONS["padding"], pady=DIMENSIONS["padding"])
        
        # Header with title
        header_frame = ttk.Frame(doc_container, style="TFrame")
        header_frame.pack(fill="x", pady=(0, DIMENSIONS["padding"]))
        
        title_label = ttk.Label(
            header_frame,
            text="Gestion des Documents",
            style="Title.TLabel"
        )
        title_label.pack(side="left", anchor="w")
        
        # Add a separator
        ttk.Separator(doc_container, orient="horizontal").pack(fill="x", pady=DIMENSIONS["padding_small"])
        
        # Description text
        description_frame = ttk.Frame(doc_container, style="TFrame")
        description_frame.pack(fill="x", pady=DIMENSIONS["padding_small"])
        
        description_label = ttk.Label(
            description_frame,
            text="Ajoutez vos documents à indexer pour le système RAG. Formats supportés : PDF, TXT",
            wraplength=600,
            style="TLabel"
        )
        description_label.pack(anchor="w", pady=(0, DIMENSIONS["padding"]))
        
        # Create a card for document actions
        action_card = ttk.Frame(doc_container, style="Card.TFrame")
        action_card.pack(fill="x", pady=DIMENSIONS["padding_small"])
        
        # Create frame for upload buttons
        buttons_frame = ttk.Frame(action_card, style="TFrame")
        buttons_frame.pack(fill="x", padx=DIMENSIONS["padding"], pady=DIMENSIONS["padding"])
        
        # Upload button with descriptive text
        upload_frame = ttk.Frame(buttons_frame, style="TFrame")
        upload_frame.pack(side="left", padx=(0, DIMENSIONS["padding_large"]))
        
        upload_title = ttk.Label(
            upload_frame,
            text="1. Ajouter des documents",
            style="Subtitle.TLabel"
        )
        upload_title.pack(anchor="w", pady=(0, 5))
        
        self.upload_button = ttk.Button(
            upload_frame,
            text="Importer un Document",
            command=self.upload_document
        )
        self.upload_button.pack(anchor="w")
        
        # Index button with descriptive text
        index_frame = ttk.Frame(buttons_frame, style="TFrame")
        index_frame.pack(side="left")
        
        index_title = ttk.Label(
            index_frame,
            text="2. Indexer les documents",
            style="Subtitle.TLabel"
        )
        index_title.pack(anchor="w", pady=(0, 5))
        
        self.index_button = ttk.Button(
            index_frame,
            text="Créer l'Index",
            command=self.index_documents,
            style="Success.TButton"
        )
        self.index_button.pack(anchor="w")
        
        # Create a card for document list
        list_card = ttk.Frame(doc_container, style="Card.TFrame")
        list_card.pack(fill="both", expand=True, pady=DIMENSIONS["padding"])
        
        list_header = ttk.Frame(list_card, style="TFrame")
        list_header.pack(fill="x", padx=DIMENSIONS["padding"], pady=DIMENSIONS["padding_small"])
        
        docs_list_label = ttk.Label(
            list_header,
            text="Documents disponibles",
            style="Subtitle.TLabel"
        )
        docs_list_label.pack(side="left")
        
        # Refresh button
        self.refresh_button = ttk.Button(
            list_header,
            text="Actualiser",
            command=self.refresh_document_list,
            width=10
        )
        self.refresh_button.pack(side="right")
        
        # Create document list with scrollbar in a card-like container
        docs_list_container = ttk.Frame(list_card, style="TFrame")
        docs_list_container.pack(fill="both", expand=True, padx=DIMENSIONS["padding"], 
                               pady=(0, DIMENSIONS["padding"]))
        
        # Create a modern-looking listbox
        self.docs_list = tk.Listbox(
            docs_list_container,
            bg=COLORS["surface"],
            fg=COLORS["text"],
            selectbackground=COLORS["primary_light"],
            selectforeground="white",
            font=FONTS["body"],
            height=15,
            borderwidth=1,
            relief="solid",
            highlightthickness=1,
            highlightcolor=COLORS["primary_light"],
            highlightbackground=COLORS["border"]
        )
        
        docs_scrollbar = ttk.Scrollbar(docs_list_container, orient="vertical", command=self.docs_list.yview)
        self.docs_list.configure(yscrollcommand=docs_scrollbar.set)
        
        self.docs_list.pack(side="left", fill="both", expand=True)
        docs_scrollbar.pack(side="right", fill="y")
        
        # Button frame for action buttons
        button_frame = ttk.Frame(list_card, style="TFrame")
        button_frame.pack(fill="x", padx=DIMENSIONS["padding"], pady=(0, DIMENSIONS["padding"]))
        
        # Add delete button
        self.delete_button = ttk.Button(
            button_frame,
            text="Supprimer",
            command=self.delete_document,
            style="Accent.TButton"
        )
        self.delete_button.pack(side="left", padx=(0, DIMENSIONS["padding"]))
        
        # Load current documents
        self.refresh_document_list()

    def setup_settings_panel(self):
        """Set up the settings tab"""
        # Create main container
        settings_container = ttk.Frame(self.settings_tab, style="TFrame")
        settings_container.pack(fill="both", expand=True, padx=DIMENSIONS["padding"], pady=DIMENSIONS["padding"])
        
        # Header with title
        header_frame = ttk.Frame(settings_container, style="TFrame")
        header_frame.pack(fill="x", pady=(0, DIMENSIONS["padding"]))
        
        title_label = ttk.Label(
            header_frame,
            text="Paramètres",
            style="Title.TLabel"
        )
        title_label.pack(side="left", anchor="w")
        
        # Add a separator
        ttk.Separator(settings_container, orient="horizontal").pack(fill="x", pady=DIMENSIONS["padding_small"])
        
        # Create a card for settings
        settings_card = ttk.Frame(settings_container, style="Card.TFrame")
        settings_card.pack(fill="both", expand=True, pady=DIMENSIONS["padding_small"])
        
        # Create scrollable content area for settings
        settings_scroll = ScrollableFrame(settings_card)
        settings_scroll.pack(fill="both", expand=True, padx=DIMENSIONS["padding"], pady=DIMENSIONS["padding"])
        
        settings_frame = settings_scroll.scrollable_frame
        
        # Section: LLM Configuration
        llm_section = ttk.Frame(settings_frame, style="TFrame")
        llm_section.pack(fill="x", pady=DIMENSIONS["padding"])
        
        llm_title = ttk.Label(
            llm_section,
            text="Configuration du Modèle",
            style="Subtitle.TLabel"
        )
        llm_title.pack(anchor="w", pady=(0, DIMENSIONS["padding"]))
        
        # LLM Provider selection - with modern styling
        llm_frame = ttk.Frame(llm_section, style="TFrame")
        llm_frame.pack(fill="x", pady=DIMENSIONS["padding_small"])
        
        llm_label = ttk.Label(llm_frame, text="Fournisseur LLM:", width=20)
        llm_label.pack(side="left", padx=(0, DIMENSIONS["padding"]))
        
        self.llm_provider_var = tk.StringVar(value=LLM_PROVIDER)
        llm_options = ["ollama", "openai"]
        llm_dropdown = ttk.Combobox(
            llm_frame,
            textvariable=self.llm_provider_var,
            values=llm_options,
            state="readonly",
            width=30
        )
        llm_dropdown.pack(side="left")
        
        # Model selection - with modern styling
        model_frame = ttk.Frame(llm_section, style="TFrame")
        model_frame.pack(fill="x", pady=DIMENSIONS["padding_small"])
        
        model_label = ttk.Label(model_frame, text="Modèle:", width=20)
        model_label.pack(side="left", padx=(0, DIMENSIONS["padding"]))
        
        self.model_var = tk.StringVar(value=LLM_MODEL if LLM_PROVIDER == "ollama" else OPENAI_MODEL)
        self.model_entry = ttk.Entry(model_frame, textvariable=self.model_var, width=32)
        self.model_entry.pack(side="left")
        
        # Separator
        ttk.Separator(settings_frame, orient="horizontal").pack(fill="x", pady=DIMENSIONS["padding"])
        
        # Section: API Configuration
        api_section = ttk.Frame(settings_frame, style="TFrame")
        api_section.pack(fill="x", pady=DIMENSIONS["padding"])
        
        api_title = ttk.Label(
            api_section,
            text="Configuration de l'API",
            style="Subtitle.TLabel"
        )
        api_title.pack(anchor="w", pady=(0, DIMENSIONS["padding"]))
        
        # API Key for OpenAI - with modern styling
        api_frame = ttk.Frame(api_section, style="TFrame")
        api_frame.pack(fill="x", pady=DIMENSIONS["padding_small"])
        
        api_label = ttk.Label(api_frame, text="Clé API OpenAI:", width=20)
        api_label.pack(side="left", padx=(0, DIMENSIONS["padding"]))
        
        self.api_key_var = tk.StringVar(value=os.getenv("OPENAI_API_KEY", ""))
        self.api_key_entry = ttk.Entry(api_frame, textvariable=self.api_key_var, width=32, show="*")
        self.api_key_entry.pack(side="left")
        
        # Show/hide password button
        self.show_key = tk.BooleanVar(value=False)
        self.show_key_button = ttk.Button(
            api_frame, 
            text="Afficher", 
            command=self.toggle_key_visibility,
            width=8
        )
        self.show_key_button.pack(side="left", padx=(DIMENSIONS["padding_small"], 0))
        
        # Separator
        ttk.Separator(settings_frame, orient="horizontal").pack(fill="x", pady=DIMENSIONS["padding"])
        
        # Section: System Prompt
        prompt_section = ttk.Frame(settings_frame, style="TFrame")
        prompt_section.pack(fill="x", pady=DIMENSIONS["padding"])
        
        prompt_title = ttk.Label(
            prompt_section,
            text="Prompt Système",
            style="Subtitle.TLabel"
        )
        prompt_title.pack(anchor="w", pady=(0, DIMENSIONS["padding"]))
        
        prompt_description = ttk.Label(
            prompt_section,
            text="Ce texte sera utilisé comme instruction pour le modèle de langage.",
            wraplength=600
        )
        prompt_description.pack(anchor="w", pady=(0, DIMENSIONS["padding"]))
        
        # System prompt with modern styling
        prompt_frame = ttk.Frame(prompt_section, style="TFrame")
        prompt_frame.pack(fill="x", pady=DIMENSIONS["padding_small"])
        
        self.system_prompt_var = tk.StringVar(value=SYSTEM_PROMPT)
        self.system_prompt_text = scrolledtext.ScrolledText(
            prompt_frame, 
            height=5, 
            width=40, 
            font=FONTS["body"],
            wrap=tk.WORD,
            bg=COLORS["surface"],
            highlightbackground=COLORS["border"],
            highlightcolor=COLORS["primary_light"],
            relief="solid",
            borderwidth=1,
            highlightthickness=1
        )
        self.system_prompt_text.insert("1.0", SYSTEM_PROMPT)
        self.system_prompt_text.pack(fill="both", expand=True)
        
        # Save button section
        save_frame = ttk.Frame(settings_frame, style="TFrame")
        save_frame.pack(fill="x", pady=DIMENSIONS["padding_large"])
        
        self.save_button = ttk.Button(
            save_frame,
            text="Enregistrer",
            command=self.save_settings,
            style="Success.TButton",
            width=15
        )
        self.save_button.pack(side="right")
        
        # Reset button
        self.reset_button = ttk.Button(
            save_frame,
            text="Réinitialiser",
            command=self.reset_settings,
            width=15
        )
        self.reset_button.pack(side="right", padx=(0, DIMENSIONS["padding"]))
        
        # Connect llm provider change to update UI
        self.llm_provider_var.trace_add("write", self.on_llm_provider_change)
        
        # Initial UI update
        self.on_llm_provider_change()

    def toggle_key_visibility(self):
        """Toggle API key visibility"""
        if self.show_key.get():
            self.api_key_entry.configure(show="*")
            self.show_key_button.configure(text="Afficher")
            self.show_key.set(False)
        else:
            self.api_key_entry.configure(show="")
            self.show_key_button.configure(text="Masquer")
            self.show_key.set(True)

    def reset_settings(self):
        """Reset settings to default values"""
        # Ask for confirmation
        if not tk.messagebox.askyesno("Confirmation", "Êtes-vous sûr de vouloir réinitialiser les paramètres?"):
            return
        
        # Reset values
        self.llm_provider_var.set("ollama")
        self.model_var.set("mistral")
        self.api_key_var.set("")
        self.system_prompt_text.delete("1.0", tk.END)
        self.system_prompt_text.insert("1.0", "You are a helpful AI assistant.")
        
        # Update UI
        self.on_llm_provider_change()
        
        # Show confirmation
        tk.messagebox.showinfo("Réinitialisation", "Les paramètres ont été réinitialisés aux valeurs par défaut.")

    def on_llm_provider_change(self, *args):
        """Update UI based on selected LLM provider"""
        provider = self.llm_provider_var.get()
        
        if provider == "ollama":
            self.model_var.set(LLM_MODEL)
            self.api_key_entry.configure(state="disabled")
            self.show_key_button.configure(state="disabled")
        else:
            self.model_var.set(OPENAI_MODEL)
            self.api_key_entry.configure(state="normal")
            self.show_key_button.configure(state="normal")

    def refresh_document_list(self):
        """Refresh the list of documents"""
        self.docs_list.delete(0, tk.END)
        
        if not os.path.exists(DATA_DIR):
            return
        
        # List files in data directory
        files = sorted([f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))])
        
        for file in files:
            self.docs_list.insert(tk.END, file)

    def delete_document(self):
        """Delete the selected document"""
        selected_indices = self.docs_list.curselection()
        if not selected_indices:
            return
        
        # Confirm deletion
        selected_files = [self.docs_list.get(i) for i in selected_indices]
        if len(selected_files) == 1:
            confirm = tk.messagebox.askyesno(
                "Confirmation de suppression",
                f"Êtes-vous sûr de vouloir supprimer '{selected_files[0]}'?"
            )
        else:
            confirm = tk.messagebox.askyesno(
                "Confirmation de suppression",
                f"Êtes-vous sûr de vouloir supprimer {len(selected_files)} fichiers?"
            )
        
        if not confirm:
            return
        
        # Delete files
        for file_name in selected_files:
            file_path = os.path.join(DATA_DIR, file_name)
            try:
                os.remove(file_path)
                logging.info(f"Deleted {file_path}")
            except Exception as e:
                logging.error(f"Error deleting file {file_path}: {e}")
                tk.messagebox.showerror(
                    "Erreur de suppression",
                    f"Erreur lors de la suppression de {file_name}: {str(e)}"
                )
        
        # Refresh document list
        self.refresh_document_list()

    def index_documents(self):
        """Index documents using the index_documents.py script"""
        # Check if documents exist
        if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
            tk.messagebox.showinfo(
                "Aucun document",
                "Aucun document trouvé à indexer. Veuillez importer des documents d'abord."
            )
            return
        
        # Ask for confirmation
        confirm = tk.messagebox.askyesno(
            "Confirmation",
            "Vous êtes sur le point d'indexer les documents. Cette opération peut prendre du temps selon le nombre et la taille des documents. Continuer?"
        )
        
        if not confirm:
            return
        
        # Start indexing in a separate thread
        threading.Thread(target=self._run_indexing, daemon=True).start()

    def _run_indexing(self):
        """Run the indexing process"""
        try:
            # Create a progress window
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Indexation des documents")
            progress_window.geometry("500x200")
            progress_window.transient(self.root)
            progress_window.grab_set()
            progress_window.configure(bg=COLORS["background"])
            
            # Center the window
            progress_window.update_idletasks()
            width = progress_window.winfo_width()
            height = progress_window.winfo_height()
            x = (self.root.winfo_width() // 2) - (width // 2) + self.root.winfo_x()
            y = (self.root.winfo_height() // 2) - (height // 2) + self.root.winfo_y()
            progress_window.geometry(f"+{x}+{y}")
            
            # Header
            header_label = tk.Label(
                progress_window, 
                text="Indexation des documents en cours", 
                bg=COLORS["background"],
                fg=COLORS["primary"],
                font=FONTS["subtitle"],
                pady=10
            )
            header_label.pack(pady=10)
            
            # Info text
            info_text = "Cette opération peut prendre plusieurs minutes selon la taille et le nombre de documents."
            info_label = tk.Label(
                progress_window, 
                text=info_text, 
                bg=COLORS["background"],
                fg=COLORS["text"],
                wraplength=450,
                justify=tk.CENTER,
                font=FONTS["body"]
            )
            info_label.pack(pady=5)
            
            # Progress animation (indeterminate)
            progress = ttk.Progressbar(
                progress_window, 
                orient="horizontal", 
                length=450, 
                mode="indeterminate"
            )
            progress.pack(pady=15, padx=20)
            progress.start(10)  # Start animation
            
            # Status label
            status_label = tk.Label(
                progress_window, 
                text="Démarrage de l'indexation...", 
                bg=COLORS["background"],
                fg=COLORS["text_secondary"],
                font=FONTS["small"]
            )
            status_label.pack(pady=5)
            
            # Cancel button
            cancel_button = ttk.Button(
                progress_window,
                text="Annuler",
                command=progress_window.destroy
            )
            cancel_button.pack(pady=10)
            
            # Define a function to update status periodically
            def update_status():
                if not progress_window.winfo_exists():
                    return
                
                statuses = [
                    "Chargement des documents...",
                    "Traitement du texte...",
                    "Découpage en chunks...",
                    "Génération des embeddings...",
                    "Construction de l'index FAISS...",
                    "Finalisation de l'indexation..."
                ]
                
                for status in statuses:
                    if not progress_window.winfo_exists():
                        return
                    
                    status_label.configure(text=status)
                    progress_window.update()
                    time.sleep(2)  # Show each status for 2 seconds
                
                # Loop back
                if progress_window.winfo_exists():
                    self.root.after(1000, update_status)
        
            # Start status updates
            self.root.after(1000, update_status)
            
            # Run the indexing script
            result = subprocess.run(
                ["python", "index_documents.py"],
                capture_output=True,
                text=True
            )
            
            # Destroy progress window if it still exists
            if progress_window.winfo_exists():
                progress_window.destroy()
            
            # Update UI with result
            if result.returncode == 0:
                tk.messagebox.showinfo(
                    "Indexation terminée",
                    "Les documents ont été indexés avec succès."
                )
                
                # Update index status
                self.check_index()
            else:
                tk.messagebox.showerror(
                    "Erreur d'indexation",
                    f"Erreur lors de l'indexation:\n{result.stderr}"
                )
        
        except Exception as e:
            logging.error(f"Error during indexing: {e}")
            tk.messagebox.showerror(
                "Erreur d'indexation",
                f"Erreur lors de l'indexation:\n{str(e)}"
            )

    def save_settings(self):
        """Save settings to .env file"""
        # Read current .env file
        env_path = ".env"
        
        # Create a dictionary of settings
        settings = {
            "LLM_PROVIDER": self.llm_provider_var.get(),
            "LLM_MODEL": self.model_var.get() if self.llm_provider_var.get() == "ollama" else os.getenv("LLM_MODEL", "mistral"),
            "OPENAI_MODEL": self.model_var.get() if self.llm_provider_var.get() == "openai" else os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            "OPENAI_API_KEY": self.api_key_var.get(),
            "SYSTEM_PROMPT": self.system_prompt_text.get("1.0", "end-1c").strip()
        }
        
        # Show saving indicator
        save_indicator = tk.Toplevel(self.root)
        save_indicator.title("")
        save_indicator.geometry("300x100")
        save_indicator.transient(self.root)
        save_indicator.configure(bg=COLORS["background"])
        
        # Remove window decorations
        save_indicator.overrideredirect(True)
        
        # Center the indicator
        save_indicator.update_idletasks()
        width = save_indicator.winfo_width()
        height = save_indicator.winfo_height()
        x = (self.root.winfo_width() // 2) - (width // 2) + self.root.winfo_x()
        y = (self.root.winfo_height() // 2) - (height // 2) + self.root.winfo_y()
        save_indicator.geometry(f"+{x}+{y}")
        
        # Add a label
        save_label = tk.Label(
            save_indicator, 
            text="Enregistrement des paramètres...", 
            bg=COLORS["background"],
            fg=COLORS["text"],
            font=FONTS["body_bold"]
        )
        save_label.pack(pady=20)
        
        # Force update
        save_indicator.update()
        
        # Update .env file
        try:
            with open(env_path, "r") as f:
                lines = f.readlines()
            
            # Create a new list of lines
            new_lines = []
            for line in lines:
                key = line.split("=")[0].strip() if "=" in line else ""
                if key in settings:
                    new_lines.append(f"{key}=\"{settings[key]}\"\n")
                    settings.pop(key)
                else:
                    new_lines.append(line)
            
            # Add any new settings
            for key, value in settings.items():
                if value:  # Only add if value is not empty
                    new_lines.append(f"{key}=\"{value}\"\n")
            
            # Write back to file
            with open(env_path, "w") as f:
                f.writelines(new_lines)
            
            # Destroy the indicator
            save_indicator.destroy()
            
            # Show success message
            tk.messagebox.showinfo(
                "Paramètres enregistrés",
                "Les paramètres ont été enregistrés avec succès. Certains changements peuvent nécessiter un redémarrage."
            )
            
            # Reload environment variables
            load_dotenv(override=True)
            
        except Exception as e:
            # Destroy the indicator
            save_indicator.destroy()
            
            logging.error(f"Error saving settings: {e}")
            tk.messagebox.showerror(
                "Erreur",
                f"Erreur lors de l'enregistrement des paramètres: {str(e)}"
            )

    def fade_in(self, alpha=0.0):
        """Create a fade-in animation when starting the app"""
        alpha += 0.1
        self.root.attributes("-alpha", alpha)
        if alpha < 1.0:
            self.root.after(20, lambda: self.fade_in(alpha))

    def setup_shortcuts(self):
        """Set up keyboard shortcuts for the application"""
        # Add keyboard shortcuts
        self.root.bind("<Control-q>", lambda e: self.root.quit())  # Ctrl+Q to quit
        self.root.bind("<Control-s>", lambda e: self.save_settings())  # Ctrl+S to save settings
        self.root.bind("<Control-1>", lambda e: self.show_chat_tab())  # Ctrl+1 for chat tab
        self.root.bind("<Control-2>", lambda e: self.show_documents_tab())  # Ctrl+2 for documents tab
        self.root.bind("<Control-3>", lambda e: self.show_settings_tab())  # Ctrl+3 for settings tab
        self.root.bind("<F5>", lambda e: self.refresh_document_list())  # F5 to refresh documents
        self.root.bind("<F1>", lambda e: self.show_help())  # F1 for help

    def show_help(self):
        """Display a help dialog with keyboard shortcuts"""
        help_window = tk.Toplevel(self.root)
        help_window.title("Aide et raccourcis clavier")
        help_window.geometry("500x400")
        help_window.transient(self.root)
        help_window.grab_set()
        help_window.configure(bg=COLORS["background"])
        
        # Center the window
        help_window.update_idletasks()
        width = help_window.winfo_width()
        height = help_window.winfo_height()
        x = (self.root.winfo_width() // 2) - (width // 2) + self.root.winfo_x()
        y = (self.root.winfo_height() // 2) - (height // 2) + self.root.winfo_y()
        help_window.geometry(f"+{x}+{y}")
        
        # Header
        header_label = tk.Label(
            help_window, 
            text="Aide et raccourcis clavier", 
            bg=COLORS["background"],
            fg=COLORS["primary"],
            font=FONTS["title"],
            pady=10
        )
        header_label.pack(pady=10)
        
        # Create a frame for shortcuts
        shortcut_frame = ttk.Frame(help_window, style="Card.TFrame", padding=10)
        shortcut_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Scrollable area for shortcuts
        shortcuts_scroll = ScrollableFrame(shortcut_frame)
        shortcuts_scroll.pack(fill="both", expand=True)
        
        shortcuts_container = shortcuts_scroll.scrollable_frame
        
        # Keyboard shortcuts list
        shortcuts = [
            ("Ctrl+1", "Aller à l'onglet Chat"),
            ("Ctrl+2", "Aller à l'onglet Documents"),
            ("Ctrl+3", "Aller à l'onglet Paramètres"),
            ("Ctrl+S", "Enregistrer les paramètres"),
            ("F5", "Actualiser la liste des documents"),
            ("F1", "Afficher l'aide"),
            ("Ctrl+Q", "Quitter l'application"),
            ("Entrée", "Envoyer un message"),
            ("Shift+Entrée", "Insérer un saut de ligne dans le message")
        ]
        
        # Add shortcuts to container
        for i, (key, desc) in enumerate(shortcuts):
            shortcut_row = ttk.Frame(shortcuts_container)
            shortcut_row.pack(fill="x", pady=5)
            
            # Key label
            key_label = tk.Label(
                shortcut_row,
                text=key,
                width=15,
                bg=COLORS["primary_light"],
                fg="white",
                font=FONTS["body_bold"],
                padx=10,
                pady=5
            )
            key_label.pack(side="left", padx=(0, 10))
            
            # Description
            desc_label = tk.Label(
                shortcut_row,
                text=desc,
                bg=COLORS["background"],
                fg=COLORS["text"],
                font=FONTS["body"],
                anchor="w"
            )
            desc_label.pack(side="left", fill="x", expand=True)
        
        # Close button
        close_button = ttk.Button(
            help_window,
            text="Fermer",
            command=help_window.destroy
        )
        close_button.pack(pady=15)

def main():
    """Run the main application"""
    root = tk.Tk()
    app = RAGChatbotApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 