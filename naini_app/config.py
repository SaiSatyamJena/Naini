# /workspace/naini_app/config.py

import os
import logging
from pathlib import Path

# --- Project Structure ---
# Assumes the script importing this config is run from /workspace OR naini_app is in PYTHONPATH
WORKSPACE_DIR = Path("/workspace") # Project root
DATA_DIR = WORKSPACE_DIR / "data"
APP_DIR = WORKSPACE_DIR / "naini_app"
GRADIO_APP_DIR = WORKSPACE_DIR # Where naini_gradio_app.py will reside

# --- Cache Directories ---
HF_CACHE_DIR = WORKSPACE_DIR / "hf-cache"
PIP_CACHE_DIR = WORKSPACE_DIR / "pip-cache"
RAG_CACHE_DIR = DATA_DIR / "rag_cache" # Directory for FAISS index, etc.

# --- Input/Output Paths ---
DEFAULT_INPUT_PDF_DIR = DATA_DIR / "persistent_pdfs"
DEFAULT_PDF_NAME = "Copy of report for FS 2020-21.pdf"
DEFAULT_PDF_PATH = DEFAULT_INPUT_PDF_DIR / DEFAULT_PDF_NAME

DEFAULT_OUTPUT_MARKDOWN_DIR = DATA_DIR / "output_markdown"
# --- ADD THIS ---
# Define a specific subdirectory and filename for the processed default PDF's markdown
# This allows us to check if it has been processed before.
DEFAULT_PDF_OUTPUT_SUBDIR = "default_processed"
DEFAULT_PDF_MARKDOWN_FILENAME = f"{DEFAULT_PDF_PATH.stem}_output.md"
DEFAULT_PDF_OUTPUT_MARKDOWN_PATH = DEFAULT_OUTPUT_MARKDOWN_DIR / DEFAULT_PDF_OUTPUT_SUBDIR / DEFAULT_PDF_MARKDOWN_FILENAME
# ---------------

DEFAULT_PERSISTENT_IMAGE_DIR = DATA_DIR / "persistent_images"
DEFAULT_PERSISTENT_SINGLE_PDF_DIR = DATA_DIR / "persistent_single_pdfs"
# --- Persistent Storage for User Uploads ---
PERSISTENT_USER_UPLOAD_DIR = WORKSPACE_DIR / "User_uploaded_pdf_md"
# --- Temporary Storage for User Uploads ---
TEMP_UPLOADS_DIR = DATA_DIR / "temp_uploads"
TEMP_PDF_DIR = TEMP_UPLOADS_DIR / "pdfs"         # Where uploaded PDFs are saved
TEMP_MARKDOWN_DIR = TEMP_UPLOADS_DIR / "markdown" # Where generated Markdown for uploads is saved/cached
TEMP_INDEX_DIR = TEMP_UPLOADS_DIR / "indices"     # Where FAISS indices for uploads are saved/cached

# --- UI Assets ---
NAINI_PIC_PATH = WORKSPACE_DIR / "Naini_pic.png"

# --- Naini Core Processing ---
IMAGE_DPI = 300 # DPI for pdf2image rendering
# Coordinate filtering margins (as percentages)
HEADER_FOOTER_TOP_MARGIN_PERCENT = 0.05
HEADER_FOOTER_BOTTOM_MARGIN_PERCENT = 0.05
# Pages designated for PyMuPDF ToC/LoT parsing (1-based index)
TOC_LIKE_PAGES = list(range(2, 8)) # Pages 2 through 7

# --- LLM Configuration ---
# Base LLM for RAG
LLM_MODEL_ID = "Qwen/Qwen2-7B-Instruct"
logging.info(f"Using LLM: {LLM_MODEL_ID}")

# Quantization (Using 4-bit as in Boe)
LLM_QUANTIZATION = "4bit" # Options: "4bit", "8bit", "none"

# LLM Generation Parameters
MAX_NEW_TOKENS = 1024  # Max tokens the LLM can generate
TEMPERATURE = 0.1      # Generation temperature (lower = more deterministic)
TOP_P = 0.95           # Nucleus sampling probability
DO_SAMPLE = True       # Whether to use sampling; False uses greedy decoding

# --- RAG Configuration (Adapted for Markdown Input) ---
# Embedding Model
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384 # Dimension of MiniLM-L6-v2
DEVICE_MAP = "auto" # For LLM and potentially embedding model if using GPU

# Chunking Strategy for Markdown
# Using MarkdownHeaderTextSplitter potentially, define headers to split on
MARKDOWN_HEADERS_TO_SPLIT_ON = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]
# Fallback chunking if Markdown splitting isn't sufficient or for non-header sections
# These values might need tuning based on the Markdown structure Naini produces
CHUNK_SIZE = 800 # Target size for text chunks
CHUNK_OVERLAP = 100 # Overlap between chunks

# Retrieval Parameters
RETRIEVER_TOP_K = 5 # Number of chunks to retrieve

# RAG Cache Naming (Example)
FAISS_INDEX_NAME = "naini_markdown_index" # Name for the FAISS index file/folder

# RAG System Prompt (Reusing Boe's geologist prompt)
RAG_SYSTEM_PROMPT = """You are Naini, an expert AI geologist assistant. Your goal is to answer questions accurately based ONLY on the provided context extracted from the processed geological Markdown document.

Follow these instructions strictly:
1.  Base your answers *solely* on the information given in the 'Relevant Context from Document' section. Do not use any prior knowledge or external information.
2.  If the context does not contain the answer to the question, state clearly that the information is not available in the provided document excerpts. Do not attempt to guess or infer information not present.
3.  When extracting information, especially from tables within the Markdown, be precise. If asked about specific values (like mineral concentrations), retrieve them exactly as presented in the context.
4.  When possible, cite the source of your information by referring to the approximate section or table mentioned in the context provided (e.g., "According to the section discussing hydrogeology..." or "Based on the table showing chemical analysis results...").
5.  Focus on retrieving and synthesizing information from the provided Markdown context.
6.  Be concise and informative in your responses.
"""
QWEN2_CHAT_TEMPLATE = """<|im_start|>system
{system_prompt}
Relevant Context from Document:
---
{context}
---<|im_end|>
{history_placeholder}<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""
# LLM Generation Parameters (Aligned closer to Boe's successful settings)
MAX_NEW_TOKENS = 512      # Boe used 512
TEMPERATURE = 0.6        # Boe used 0.6
TOP_P = 0.9              # Boe used 0.9
DO_SAMPLE = True         # Boe used True
# --- UI Configuration ---
GRADIO_SERVER_NAME = "0.0.0.0"
GRADIO_SERVER_PORT = 7860
APP_TITLE = "Naini - Agentic Geology Assistant"
CREATOR_NAME = "Wizard Sai" # Customize as needed

# UI Status Messages
STATUS_IDLE = "Idle"
STATUS_PROCESSING = "Processing..."
STATUS_COMPLETE = "Complete"
STATUS_ERROR = "Error"
STATUS_READY = "Ready"
STATUS_LOADING_LLM = "Loading Language Model..."
STATUS_INDEXING = "Indexing Document..."
STATUS_GENERATING = "Generating Response..."

# --- Environment Setup for Caching (Optional but recommended) ---
os.environ['TRANSFORMERS_CACHE'] = str(HF_CACHE_DIR / "transformers")
os.environ['HF_HOME'] = str(HF_CACHE_DIR / "hub")
os.environ['HF_DATASETS_CACHE'] = str(HF_CACHE_DIR / "datasets")
logging.info(f"Hugging Face Hub cache set to: {os.environ['HF_HOME']}")

# --- Logging ---
# Basic logging setup, can be overridden in main scripts
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s')