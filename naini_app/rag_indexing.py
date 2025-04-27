# /workspace/naini_app/rag_indexing.py

import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple

# Langchain components
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_core.documents import Document

# Naini config
from . import config # Use relative import within the package

log = logging.getLogger(__name__)

# --- Helper Functions ---

def load_markdown_document(markdown_path: Path) -> Optional[str]:
    """Loads the content of the Markdown file."""
    if not markdown_path.is_file():
        log.error(f"Markdown file not found: {markdown_path}")
        return None
    try:
        content = markdown_path.read_text(encoding='utf-8')
        log.info(f"Successfully loaded Markdown file: {markdown_path} ({len(content)} chars)")
        return content
    except Exception as e:
        log.error(f"Error reading Markdown file {markdown_path}: {e}", exc_info=True)
        return None

# --- MODIFIED FUNCTION ---
def split_markdown(markdown_content: str, source_filename: str) -> List[Document]:
    """
    Splits Markdown content into documents, trying header splitting first.

    Args:
        markdown_content: The raw markdown text.
        source_filename: The name of the original file (for metadata).

    Returns:
        A list of Langchain Document objects (chunks).
    """
    if not markdown_content:
        return []

    chunks = []
    try:
        log.info("Attempting Markdown splitting by headers...")
        headers_to_split_on = config.MARKDOWN_HEADERS_TO_SPLIT_ON
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            return_each_line=False
        )
        chunks = markdown_splitter.split_text(markdown_content)
        log.info(f"Split into {len(chunks)} chunks using MarkdownHeaderTextSplitter.")

        if len(chunks) <= 5: # Threshold might need tuning
             log.warning(f"Markdown header splitting produced only {len(chunks)} chunks. Falling back to RecursiveCharacterTextSplitter for potentially better granularity.")
             chunks = []

    except Exception as e:
        log.warning(f"MarkdownHeaderTextSplitter failed: {e}. Falling back to RecursiveCharacterTextSplitter.", exc_info=False)
        chunks = []

    if not chunks:
        log.info("Using RecursiveCharacterTextSplitter...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            add_start_index=True,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        temp_doc = Document(page_content=markdown_content)
        chunks = text_splitter.split_documents([temp_doc])
        log.info(f"Split into {len(chunks)} chunks using RecursiveCharacterTextSplitter.")

    # Add common metadata (source file) to all chunks
    for chunk in chunks:
        # Ensure metadata dict exists
        if chunk.metadata is None: chunk.metadata = {}
        # Add/update source filename
        chunk.metadata["source"] = source_filename # <--- MODIFIED: Use parameter

    return chunks


# --- Main Indexing Function ---

# --- MODIFIED FUNCTION ---
def build_and_save_faiss_index(
    markdown_path: Path,
    embedding_model: HuggingFaceEmbeddings,
    index_folder_path: Path, # NEW: Specify exact folder path
    index_name: str,         # NEW: Specify exact index name
    force_recreate: bool = False
) -> Optional[FAISS]:
    """
    Loads Markdown, chunks it, creates embeddings, builds a FAISS index,
    and saves it to the specified path and name.

    Args:
        markdown_path: Path to the processed Markdown file.
        embedding_model: The loaded HuggingFaceEmbeddings model instance.
        index_folder_path: The directory where the index files should be saved/loaded from.
        index_name: The base name for the index files (e.g., 'my_document_index').
        force_recreate: If True, ignore existing index and rebuild.

    Returns:
        The loaded or newly created FAISS vector store instance, or None on failure.
    """
    log.info(f"--- Starting RAG Index Building for {markdown_path.name} ---")
    # --- MODIFIED: Use parameters for path ---
    full_index_path_prefix = index_folder_path / index_name
    log.info(f"Target index path prefix: {full_index_path_prefix}")

    # --- Cache Check ---
    # Check if BOTH .faiss and .pkl files exist for a valid load
    faiss_file = index_folder_path / f"{index_name}.faiss"
    pkl_file = index_folder_path / f"{index_name}.pkl"

    if not force_recreate and faiss_file.exists() and pkl_file.exists():
        try:
            log.info(f"Found existing FAISS index files for '{index_name}' in {index_folder_path}. Loading...")
            vector_store = FAISS.load_local(
                folder_path=str(index_folder_path), # MODIFIED: Use parameter
                embeddings=embedding_model,
                index_name=index_name,              # MODIFIED: Use parameter
                allow_dangerous_deserialization=True
            )
            log.info(f"Successfully loaded FAISS index '{index_name}' from {index_folder_path}.")
            return vector_store
        except Exception as e:
            log.warning(f"Error loading existing FAISS index '{index_name}' from {index_folder_path}: {e}. Recreating...", exc_info=True)
    elif force_recreate:
         log.info(f"Force_recreate is True. Rebuilding index '{index_name}'...")
    else:
         log.info(f"No existing index found for '{index_name}' at {index_folder_path}. Creating new index...")

    # --- Load and Chunk Markdown ---
    markdown_content = load_markdown_document(markdown_path)
    if not markdown_content:
        return None # Error already logged

    log.info("Splitting Markdown document...")
    start_chunk_time = time.time()
    # --- MODIFIED CALL: Pass source filename ---
    documents_to_index = split_markdown(markdown_content, markdown_path.name)
    # ----------------------------------------
    end_chunk_time = time.time()
    log.info(f"Splitting done ({len(documents_to_index)} chunks) in {end_chunk_time - start_chunk_time:.2f}s.")

    if not documents_to_index:
        log.error("No document chunks generated after splitting. Cannot build index.")
        return None

    # --- Create FAISS Index ---
    log.info(f"Creating FAISS index '{index_name}' from {len(documents_to_index)} chunks using '{config.EMBEDDING_MODEL_ID}'...")
    start_index_time = time.time()
    try:
        # Ensure cache directory exists
        index_folder_path.mkdir(parents=True, exist_ok=True) # MODIFIED: Use parameter

        # Create index from documents
        vector_store = FAISS.from_documents(documents_to_index, embedding_model)
        end_index_time = time.time()
        log.info(f"FAISS index '{index_name}' creation successful in {end_index_time - start_index_time:.2f}s.")

        # --- Save Index ---
        try:
            vector_store.save_local(
                folder_path=str(index_folder_path), # MODIFIED: Use parameter
                index_name=index_name               # MODIFIED: Use parameter
            )
            log.info(f"Successfully saved FAISS index '{index_name}' to {index_folder_path}")
            return vector_store
        except Exception as e_save:
            log.error(f"Failed to save FAISS index '{index_name}' to {index_folder_path}: {e_save}", exc_info=True)
            return vector_store # Return in-memory store even if saving failed

    except Exception as e_create:
        log.error(f"Error creating FAISS index '{index_name}': {e_create}", exc_info=True)
        return None

# --- Example Usage Block (for testing this module standalone) ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG) # Set to DEBUG for detailed logs
    log.info("--- Testing RAG Indexing Standalone ---")

    # Ensure config points to the right place
    markdown_file_path = config.DEFAULT_PDF_OUTPUT_MARKDOWN_PATH
    if not markdown_file_path.is_file():
        print(f"ERROR: Default Markdown file not found at {markdown_file_path}")
        print("Please run the main Gradio app first or the modified main.py standalone.")
        exit()
    
    # Define index path/name for testing (using defaults from config)
    test_index_folder = config.RAG_CACHE_DIR
    test_index_name = config.FAISS_INDEX_NAME

    print(f"Using Markdown file: {markdown_file_path}")
    print(f"Using Embedding model: {config.EMBEDDING_MODEL_ID}")
    # print(f"Index will be saved to: {config.RAG_CACHE_DIR / config.FAISS_INDEX_NAME}")
    print(f"Index will be saved to folder: {test_index_folder}")

    # Load embedding model (similar to Gradio app)
    embedding_instance = None
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        cache_folder = str(config.HF_CACHE_DIR / "sentence_transformers")
        embedding_instance = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL_ID,
            cache_folder=cache_folder,
            model_kwargs={'device': device}
        )
        log.info(f"Test: Embedding model loaded on {device}.")
    except Exception as e:
        print(f"ERROR: Failed to load embedding model for test: {e}")
        exit()

    # Build the index (force recreate for testing)
    print("\nBuilding index (force_recreate=True)...")
    start_build_time = time.time()
    # --- MODIFIED CALL ---
    vector_store_instance = build_and_save_faiss_index(
        markdown_path=markdown_file_path,
        embedding_model=embedding_instance,
        index_folder_path=test_index_folder, # Pass folder path
        index_name=test_index_name,           # Pass index name
        force_recreate=True
    )
    # --------------------
    end_build_time = time.time()
    print(f"Index building process finished in {end_build_time - start_build_time:.2f} seconds.")

    if vector_store_instance:
        print("Index built successfully.")
        print(f"Index contains {vector_store_instance.index.ntotal} vectors.")

        # --- Test Similarity Search ---
        print("\n--- Testing Similarity Search ---")
        # (Keep the existing similarity search test logic)
        test_query = "What is the hydrogeology of the area?"
        print(f"Searching for: '{test_query}'")
        try:
            search_start_time = time.time()
            results = vector_store_instance.similarity_search(test_query, k=3) # Get top 3 results
            search_end_time = time.time()
            print(f"Search completed in {search_end_time - search_start_time:.2f}s.")

            if results:
                print(f"\nFound {len(results)} relevant chunks:")
                for i, doc in enumerate(results):
                    source = doc.metadata.get('source', 'N/A')
                    header_info = ""
                    # Check for header metadata added by MarkdownHeaderTextSplitter
                    for h_key in config.MARKDOWN_HEADERS_TO_SPLIT_ON:
                        if h_key[1] in doc.metadata:
                             header_info += f" {h_key[1]}: '{doc.metadata[h_key[1]]}'"
                    if not header_info and 'start_index' in doc.metadata:
                         header_info = f" (Approx. start char: {doc.metadata['start_index']})"

                    print(f"\n--- Chunk {i+1} ---")
                    print(f"Source: {source}{header_info}")
                    print(f"Content: {doc.page_content[:400]}...") # Show preview
            else:
                print("No relevant chunks found for the test query.")
        except Exception as search_e:
            print(f"Error during similarity search test: {search_e}")

    else:
        print("Index building failed. Check logs.")

    print("\n--- Standalone Test Finished ---")