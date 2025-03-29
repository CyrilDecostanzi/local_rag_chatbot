#!/usr/bin/env python3
"""
Web interface for the RAG chatbot using Gradio.
This module provides a modern, responsive web interface.
"""
import os
import sys
import time
import logging
import shutil
import threading
from pathlib import Path

import gradio as gr

from src.core.llm import ask_llm
from src.core.indexing import run_indexing
from src.config.settings import (
    COLORS,
    LLM_PROVIDER,
    LLM_MODEL,
    OPENAI_MODEL,
    SYSTEM_PROMPT,
    DATA_DIR,
    INDEX_DIR,
    OPENAI_API_KEY,
    DEFAULT_TOP_K,
    RETRIEVAL_MULTIPLIER,
    USE_RERANKING,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

# Log configuration
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

# Define custom theme
custom_theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_sm,
    text_size=gr.themes.sizes.text_md,
).set(
    button_primary_background_fill=COLORS["primary"],
    button_primary_background_fill_hover=COLORS["primary_light"],
    button_secondary_background_fill=COLORS["secondary"],
    button_secondary_background_fill_hover=COLORS["secondary_hover"],
    background_fill_primary=COLORS["background"],
    block_title_text_color=COLORS["primary"],
    block_label_text_color=COLORS["text_secondary"],
    input_background_fill=COLORS["surface"],
)

def check_index_status():
    """Check if the FAISS index exists and return status info"""
    index_path = os.path.join(INDEX_DIR, "index.faiss")
    docs_path = os.path.join(INDEX_DIR, "documents.pkl")
    
    if os.path.exists(index_path) and os.path.exists(docs_path):
        file_count = len([f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))])
        return f"✅ L'index RAG est prêt. {file_count} document(s) dans le répertoire de données."
    else:
        return "❌ L'index RAG n'existe pas. Veuillez indexer des documents."

def list_documents():
    """List all documents in the data directory"""
    if not os.path.exists(DATA_DIR):
        return []
    
    files = []
    for f in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, f)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path)
            size_str = f"{size / 1024:.1f} KB" if size < 1024 * 1024 else f"{size / (1024 * 1024):.1f} MB"
            files.append(f"{f} ({size_str})")
    
    return files

def handle_chat(message, history):
    """Process a chat message and return the response"""
    if not message:
        return "Veuillez entrer un message."
    
    # Check if index exists
    index_path = os.path.join(INDEX_DIR, "index.faiss")
    docs_path = os.path.join(INDEX_DIR, "documents.pkl")
    
    if not os.path.exists(index_path) or not os.path.exists(docs_path):
        return "⚠️ L'index RAG n'existe pas. Veuillez d'abord indexer des documents en allant dans l'onglet 'Documents'."
    
    # Get response from LLM
    response = ask_llm(message)
    return response

def handle_file_upload(files):
    """Process uploaded files and move them to the data directory"""
    if not files:
        return "Aucun fichier sélectionné.", list_documents()
    
    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    success_count = 0
    for file in files:
        try:
            filename = os.path.basename(file.name)
            destination = os.path.join(DATA_DIR, filename)
            
            # Copy file to data directory
            shutil.copy2(file.name, destination)
            success_count += 1
            logging.info(f"File uploaded: {filename}")
        except Exception as e:
            logging.error(f"Error uploading file {file.name}: {e}")
    
    return f"✅ {success_count} fichier(s) ajouté(s) avec succès.", list_documents()

def handle_file_deletion(files_to_delete):
    """Delete selected files from the data directory"""
    if not files_to_delete:
        return "Aucun fichier sélectionné pour la suppression.", list_documents()
    
    success_count = 0
    for file_info in files_to_delete:
        try:
            # Extract filename from the file info string (remove size info)
            filename = file_info.split(" (")[0]
            file_path = os.path.join(DATA_DIR, filename)
            
            if os.path.exists(file_path):
                os.remove(file_path)
                success_count += 1
                logging.info(f"File deleted: {filename}")
        except Exception as e:
            logging.error(f"Error deleting file {file_info}: {e}")
    
    return f"✅ {success_count} fichier(s) supprimé(s) avec succès.", list_documents()

def handle_indexing():
    """Run the indexing process"""
    yield "⏳ Démarrage de l'indexation des documents..."
    success = run_indexing(DATA_DIR)
    
    if success:
        yield "✅ Indexation terminée avec succès ! Vous pouvez maintenant utiliser le chatbot."
    else:
        yield "❌ Erreur lors de l'indexation. Vérifiez les logs pour plus de détails."

def handle_clear_index():
    """Clear the FAISS index"""
    index_path = os.path.join(INDEX_DIR, "index.faiss")
    docs_path = os.path.join(INDEX_DIR, "documents.pkl")
    
    try:
        if os.path.exists(index_path):
            os.remove(index_path)
        if os.path.exists(docs_path):
            os.remove(docs_path)
        return "✅ Index RAG vidé avec succès."
    except Exception as e:
        logging.error(f"Error clearing index: {e}")
        return f"❌ Erreur lors de la suppression de l'index: {str(e)}"

def save_settings(llm_provider, llm_model, openai_model, system_prompt, 
                 retrieval_k, use_reranking, chunk_size, chunk_overlap):
    """Save settings to .env file"""
    try:
        env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
        
        # Read existing .env file
        with open(env_path, 'r') as file:
            lines = file.readlines()
        
        # Update values
        updated_lines = []
        settings = {
            'LLM_PROVIDER': llm_provider,
            'LLM_MODEL': llm_model,
            'OPENAI_MODEL': openai_model,
            'SYSTEM_PROMPT': system_prompt,
            'DEFAULT_TOP_K': str(retrieval_k),
            'USE_RERANKING': str(use_reranking),
            'CHUNK_SIZE': str(chunk_size),
            'CHUNK_OVERLAP': str(chunk_overlap),
        }
        
        for line in lines:
            key = line.split('=')[0].strip() if '=' in line else None
            if key in settings:
                updated_lines.append(f"{key}={settings[key]}\n")
                del settings[key]
            else:
                updated_lines.append(line)
        
        # Add any new settings
        for key, value in settings.items():
            updated_lines.append(f"{key}={value}\n")
        
        # Write back to file
        with open(env_path, 'w') as file:
            file.writelines(updated_lines)
        
        # Force reload settings (in a real app, you might need to restart the app)
        return "✅ Paramètres sauvegardés avec succès. Certains changements pourraient nécessiter un redémarrage."
    except Exception as e:
        logging.error(f"Error saving settings: {e}")
        return f"❌ Erreur lors de la sauvegarde: {str(e)}"

def build_chat_interface():
    """Build the chat interface"""
    with gr.Blocks() as interface:
        gr.Markdown("# 💬 RAG Chatbot")
        
        with gr.Row():
            with gr.Column(scale=7):
                chatbot = gr.Chatbot(
                    height=550,
                    bubble_full_width=False,
                    avatar_images=("👤", "🤖"),
                    show_copy_button=True,
                )
                
                with gr.Row():
                    user_input = gr.Textbox(
                        placeholder="Posez votre question ici...",
                        scale=9,
                        container=False,
                        show_label=False,
                    )
                    send_btn = gr.Button("Envoyer", scale=1)
                
                status = gr.Markdown(check_index_status())
            
            with gr.Column(scale=3):
                gr.Markdown("### ⚙️ Paramètres Rapides")
                
                provider = gr.Dropdown(
                    choices=["ollama", "openai"],
                    value=LLM_PROVIDER,
                    label="Fournisseur LLM",
                )
                
                model_ollama = gr.Dropdown(
                    choices=["mistral", "llama2", "phi"],
                    value=LLM_MODEL,
                    label="Modèle (Ollama)",
                    visible=(LLM_PROVIDER=="ollama")
                )
                
                model_openai = gr.Dropdown(
                    choices=["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"],
                    value=OPENAI_MODEL,
                    label="Modèle (OpenAI)",
                    visible=(LLM_PROVIDER=="openai")
                )
                
                api_key = gr.Textbox(
                    value=OPENAI_API_KEY,
                    label="Clé API OpenAI",
                    type="password",
                    visible=(LLM_PROVIDER=="openai")
                )
                
                # Dynamic UI based on provider selection
                def update_model_visibility(provider):
                    return {
                        model_ollama: provider == "ollama",
                        model_openai: provider == "openai",
                        api_key: provider == "openai"
                    }
                
                provider.change(
                    update_model_visibility,
                    inputs=provider,
                    outputs=[model_ollama, model_openai, api_key]
                )
                
                gr.Markdown("### 🔍 Statistiques")
                
                with gr.Row():
                    clear_btn = gr.Button("Effacer la conversation")
                    refresh_btn = gr.Button("Rafraîchir le statut")
                
                refresh_btn.click(
                    check_index_status,
                    outputs=status
                )
                
                clear_btn.click(
                    lambda: None,
                    outputs=chatbot,
                    show_progress=False
                )
        
        # Handle sending messages
        def user_message(user_message, history):
            return "", history + [[user_message, None]]
        
        def bot_message(history):
            user_message = history[-1][0]
            bot_response = handle_chat(user_message, history[:-1])
            history[-1][1] = bot_response
            return history
        
        user_input.submit(
            user_message,
            [user_input, chatbot],
            [user_input, chatbot],
            queue=False
        ).then(
            bot_message,
            chatbot,
            chatbot
        )
        
        send_btn.click(
            user_message,
            [user_input, chatbot],
            [user_input, chatbot],
            queue=False
        ).then(
            bot_message,
            chatbot,
            chatbot
        )
    
    return interface

def build_documents_interface():
    """Build the documents management interface"""
    with gr.Blocks() as interface:
        gr.Markdown("# 📄 Gestion des Documents")
        
        with gr.Row():
            with gr.Column(scale=6):
                file_upload = gr.File(
                    file_count="multiple",
                    label="Ajouter des fichiers",
                    type="filepath"
                )
                
                upload_btn = gr.Button("Téléverser les fichiers", variant="primary")
                upload_status = gr.Markdown()
                
                document_list = gr.CheckboxGroup(
                    choices=list_documents(),
                    label="Documents disponibles",
                    info="Sélectionnez des documents pour les supprimer"
                )
                
                with gr.Row():
                    delete_btn = gr.Button("Supprimer la sélection", variant="stop")
                    refresh_btn = gr.Button("Rafraîchir la liste")
                
                delete_status = gr.Markdown()
            
            with gr.Column(scale=4):
                gr.Markdown("### 🔍 Indexation RAG")
                index_status = gr.Markdown(check_index_status())
                
                with gr.Row():
                    index_btn = gr.Button("Indexer les documents", variant="primary")
                    clear_index_btn = gr.Button("Vider l'index", variant="stop")
                
                indexing_status = gr.Markdown()
                
                gr.Markdown("### ℹ️ Instructions")
                gr.Markdown("""
                1. **Ajoutez** des documents PDF ou TXT
                2. **Indexez** les documents pour le système RAG
                3. Retournez à l'onglet Chat pour poser des questions
                
                Formats supportés:
                - PDF (`.pdf`)
                - Texte (`.txt`)
                
                Pour vider complètement le système et recommencer, utilisez le bouton "Vider l'index".
                """)
        
        # Handle events
        upload_btn.click(
            handle_file_upload,
            inputs=file_upload,
            outputs=[upload_status, document_list]
        )
        
        delete_btn.click(
            handle_file_deletion,
            inputs=document_list,
            outputs=[delete_status, document_list]
        )
        
        refresh_btn.click(
            list_documents,
            outputs=document_list
        )
        
        refresh_btn.click(
            check_index_status,
            outputs=index_status
        )
        
        index_btn.click(
            handle_indexing,
            outputs=indexing_status
        )
        
        clear_index_btn.click(
            handle_clear_index,
            outputs=indexing_status
        )
    
    return interface

def build_settings_interface():
    """Build the settings interface"""
    with gr.Blocks() as interface:
        gr.Markdown("# ⚙️ Paramètres Avancés")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🤖 Modèle LLM")
                
                provider = gr.Radio(
                    choices=["ollama", "openai"],
                    value=LLM_PROVIDER,
                    label="Fournisseur de LLM",
                    info="Sélectionnez le fournisseur de modèle à utiliser"
                )
                
                ollama_model = gr.Textbox(
                    value=LLM_MODEL,
                    label="Modèle Ollama",
                    info="Nom du modèle dans Ollama local (ex: mistral, llama2)",
                    visible=(LLM_PROVIDER=="ollama")
                )
                
                openai_model = gr.Textbox(
                    value=OPENAI_MODEL,
                    label="Modèle OpenAI",
                    info="Nom du modèle OpenAI à utiliser (ex: gpt-3.5-turbo)",
                    visible=(LLM_PROVIDER=="openai")
                )
                
                api_key = gr.Textbox(
                    value=OPENAI_API_KEY,
                    label="Clé API OpenAI",
                    type="password",
                    visible=(LLM_PROVIDER=="openai")
                )
                
                system_prompt = gr.Textbox(
                    value=SYSTEM_PROMPT,
                    label="Prompt Système",
                    info="Instructions données au modèle",
                    lines=3
                )
                
                # Dynamic UI based on provider selection
                def update_settings_visibility(provider):
                    return {
                        ollama_model: provider == "ollama",
                        openai_model: provider == "openai",
                        api_key: provider == "openai"
                    }
                
                provider.change(
                    update_settings_visibility,
                    inputs=provider,
                    outputs=[ollama_model, openai_model, api_key]
                )
                
                gr.Markdown("### 🔍 Paramètres de Récupération")
                
                retrieval_k = gr.Slider(
                    minimum=1,
                    maximum=10, 
                    value=DEFAULT_TOP_K,
                    step=1,
                    label="Nombre de chunks à récupérer (top-k)",
                    info="Nombre de morceaux de texte à récupérer de la base de données"
                )
                
                use_reranking = gr.Checkbox(
                    value=USE_RERANKING,
                    label="Utiliser le reranking",
                    info="Améliore la pertinence des résultats mais ralentit le traitement"
                )
                
                chunk_size = gr.Slider(
                    minimum=100,
                    maximum=2000,
                    value=CHUNK_SIZE,
                    step=50,
                    label="Taille des chunks",
                    info="Taille des morceaux de texte lors de l'indexation"
                )
                
                chunk_overlap = gr.Slider(
                    minimum=0,
                    maximum=200,
                    value=CHUNK_OVERLAP,
                    step=10,
                    label="Chevauchement des chunks",
                    info="Chevauchement entre les morceaux de texte pour assurer la continuité"
                )
                
                save_btn = gr.Button("Sauvegarder les paramètres", variant="primary")
                settings_status = gr.Markdown()
                
                save_btn.click(
                    save_settings,
                    inputs=[
                        provider, 
                        ollama_model, 
                        openai_model, 
                        system_prompt, 
                        retrieval_k, 
                        use_reranking, 
                        chunk_size, 
                        chunk_overlap
                    ],
                    outputs=settings_status
                )
            
            with gr.Column():
                gr.Markdown("### 📊 Informations")
                gr.Markdown(f"""
                **Configuration actuelle:**
                - Fournisseur LLM: `{LLM_PROVIDER}`
                - Modèle: `{LLM_MODEL if LLM_PROVIDER == 'ollama' else OPENAI_MODEL}`
                - Répertoire de données: `{DATA_DIR}`
                - Répertoire d'index: `{INDEX_DIR}`
                
                **Modèles d'embedding:**
                - Modèle principal: `all-MiniLM-L6-v2`
                - Modèle de reranking: `cross-encoder/ms-marco-MiniLM-L-6-v2`
                
                **Version du système:**
                - RAG Chatbot v1.0.0
                """)
                
                gr.Markdown("### 🛠️ Dépannage")
                gr.Markdown("""
                **Problèmes courants:**
                
                1. **Erreur de connexion à Ollama**
                   - Vérifiez qu'Ollama est en cours d'exécution sur votre machine
                   - Exécutez `ollama serve` dans un terminal
                
                2. **Erreur d'API OpenAI**
                   - Vérifiez votre clé API
                   - Vérifiez votre quota/crédits disponibles
                
                3. **Problèmes d'indexation**
                   - Vérifiez que vos documents sont dans des formats supportés
                   - Assurez-vous d'avoir des permissions d'écriture dans le dossier d'index
                """)
                
                gr.Markdown("### 🧪 Fonctionnalités expérimentales")
                gr.Markdown("""
                **À venir:**
                - Support pour les documents Word (.docx)
                - Support pour les présentations PowerPoint (.pptx)
                - Extraction d'images et de tableaux
                - Streaming des réponses
                """)
    
    return interface

def run_web_ui():
    """Main function to run the web interface"""
    # Create the Gradio interface
    with gr.Blocks(theme=custom_theme) as demo:
        with gr.Tabs():
            with gr.TabItem("Chat", id=0):
                chat_interface = build_chat_interface()
            
            with gr.TabItem("Documents", id=1):
                docs_interface = build_documents_interface()
            
            with gr.TabItem("Paramètres", id=2):
                settings_interface = build_settings_interface()
        
        # Footer
        gr.Markdown("""
        ---
        ### RAG Chatbot | Interface Web v1.0
        Recherche et interaction avec vos documents en utilisant la génération augmentée par récupération (RAG).
        """)
    
    # Launch the interface
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        share=False,
        inbrowser=True,
        show_error=True,
    )

if __name__ == "__main__":
    logging.info("Starting RAG Chatbot Web Interface...")
    run_web_ui() 