# /workspace/naini_app/rag_chain.py
# Version: Use ChatPromptTemplate.from_template with explicit Qwen2 format

import logging
import time
from pathlib import Path
from typing import Optional, Any, List, Tuple, Union
from operator import itemgetter

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
# Explicitly import ChatPromptTemplate components
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # MessagesPlaceholder kept for potential future use
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.documents import Document
# --- Added for handling chat history formatting ---
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage

# Assume config is accessible via naini_app
from naini_app import config
from naini_app.rag_indexing import build_and_save_faiss_index

log = logging.getLogger(__name__)

def format_docs_for_context(docs: List[Document]) -> str:
    """Formats retrieved Markdown chunks for the LLM context."""
    if not docs:
        return "No relevant context found in the document for this question."

    context_str = ""
    separator = "\n\n---\n\n"
    for i, doc in enumerate(docs):
        # Try to get header, fallback to source, then N/A
        source_info = doc.metadata.get('header', doc.metadata.get('source', 'N/A'))
        meta_info = f"Source Chunk {i+1}: {source_info}"
        context_str += f"Context from: {meta_info}\n"
        context_str += doc.page_content
        context_str += separator

    # Remove trailing separator and whitespace
    return context_str.strip().rstrip('-').strip()

# --- NEW FUNCTION: Format chat history for the explicit template ---
def format_chat_history_for_template(chat_history: List[BaseMessage]) -> str:
    """Formats chat history into a string compatible with Qwen2 template."""
    if not chat_history:
        return "" # Return empty string if no history

    history_str = ""
    for message in chat_history:
        if isinstance(message, HumanMessage):
            history_str += f"<|im_start|>user\n{message.content}<|im_end|>\n"
        elif isinstance(message, AIMessage):
            history_str += f"<|im_start|>assistant\n{message.content}<|im_end|>\n"
        # Ignore system messages if they accidentally end up here
    return history_str.strip() # Remove trailing newline if any

# --- MODIFIED FUNCTION SIGNATURE ---
def setup_rag_chain_components(
    markdown_path: Path,
    embedding_model: HuggingFaceEmbeddings,
    index_folder_path: Path, # NEW: Specify index folder path
    index_name: str          # NEW: Specify index name
) -> Tuple[Optional[FAISS], Optional[Any], Optional[str]]:
    """
    Loads/Builds the FAISS index using specified paths/names and constructs the
    RAG chain components up to the prompt.

    Args:
        markdown_path: Path to the processed Markdown file.
        embedding_model: The loaded HuggingFace embedding model instance.
        index_folder_path: The directory to load/save the FAISS index from/to.
        index_name: The base name for the FAISS index files.

    Returns:
        A tuple containing:
        - The loaded/built FAISS vector store instance (Optional[FAISS]).
        - The constructed RAG chain up to the prompt (Optional[Runnable]).
        - An error message string if setup failed, otherwise None (Optional[str]).
    """
    log.info(f"Setting up RAG components for '{markdown_path.name}'...")
    log.info(f"Using index: '{index_name}' in folder: {index_folder_path}")

    # --- Step 1: Load/Build Vector Store ---
    vector_store_instance: Optional[FAISS] = None
    error_message: Optional[str] = None

    log.info(f"Loading/Building Index for '{markdown_path.name}'...")
    start_index_time = time.time()
    try:
        # --- MODIFIED CALL ---
        vector_store_instance = build_and_save_faiss_index(
            markdown_path=markdown_path,
            embedding_model=embedding_model,
            index_folder_path=index_folder_path, # Pass dynamic path
            index_name=index_name,               # Pass dynamic name
            force_recreate=False # Usually False, let caller decide if needed
        )
        # --------------------
        end_index_time = time.time()
        if not vector_store_instance:
            error_message = f"Failed to build or load RAG index '{index_name}' from '{index_folder_path}'."
            log.error(error_message)
            return None, None, error_message
        else:
            log.info(f"Index '{index_name}' ready ({end_index_time - start_index_time:.2f}s). Vectors: {vector_store_instance.index.ntotal}")
    except Exception as e:
        error_message = f"Error during FAISS index build/load for '{index_name}': {e}"
        log.error(error_message, exc_info=True)
        return None, None, error_message

    # --- Step 2: Build RAG Chain (Up to Prompt) - MODIFIED ---
    rag_chain_up_to_prompt: Optional[Any] = None
    log.info("Building RAG chain components (Up to Prompt) using explicit template...")
    try:
        retriever = vector_store_instance.as_retriever(
            search_type="similarity",
            search_kwargs={"k": config.RETRIEVER_TOP_K}
        )

        # --- Use ChatPromptTemplate.from_template with the QWEN2 format ---
        template_string = getattr(config, 'QWEN2_CHAT_TEMPLATE', None)
        if not template_string:
            error_message = "QWEN2_CHAT_TEMPLATE not found in config.py"
            log.error(error_message)
            return vector_store_instance, None, error_message
        if "{history_placeholder}" not in template_string:
             # Add a simple placeholder if missing, though the one in config is preferred
             log.warning("'{history_placeholder}' not found in QWEN2_CHAT_TEMPLATE. Adding basic support.")
             # This assumes the template ends like "...<|im_end|>\n<|im_start|>user..."
             template_string = template_string.replace("<|im_start|>user", "{history_placeholder}<|im_start|>user")


        log.info("Using ChatPromptTemplate.from_template.")
        prompt = ChatPromptTemplate.from_template(template_string)
        # --------------------------------------------------------------------

        # Define the RAG setup using RunnableParallel
        # NOTE: We now need to format chat_history explicitly using RunnableLambda
        setup_and_retrieval = RunnableParallel(
            {
                "context": itemgetter("question") | retriever | RunnableLambda(format_docs_for_context),
                "question": itemgetter("question"),
                # Apply the new formatting function to the chat_history list
                "history_placeholder": itemgetter("chat_history") | RunnableLambda(format_chat_history_for_template),
                "system_prompt": RunnableLambda(lambda _input: config.RAG_SYSTEM_PROMPT), # Pass system prompt directly, ignore input <--- CORRECTED LINE
            }
        )

        # Final chain structure (up to the prompt formatting step)
        rag_chain_up_to_prompt = setup_and_retrieval | prompt

        log.info("RAG chain components (Up to Prompt) built successfully using explicit template.")
        return vector_store_instance, rag_chain_up_to_prompt, None # Success

    except Exception as e:
        error_message = f"Failed to build RAG chain components: {e}"
        log.error(error_message, exc_info=True)
        return vector_store_instance, None, error_message