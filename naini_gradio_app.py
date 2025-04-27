# /workspace/naini_gradio_app.py
# Version: V5.9 - User Uploads Implemented with Caching & UI Refresh

import hashlib
import shutil # For potential cleanup later
import re
import gradio as gr
import time
import os
import logging # Keep logging import
import traceback
from pathlib import Path
import sys
from typing import Optional, Dict, Any, Generator, Tuple, List, Union
from naini_app.layout_detector import load_layout_model_and_processor, detect_layout_elements # Import loader

# --- Imports for Models & LangChain ---
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig,
    PreTrainedTokenizer, PreTrainedTokenizerFast, TextIteratorStreamer,
    StoppingCriteria, StoppingCriteriaList
)
from threading import Thread
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_core.runnables import RunnableConfig, Runnable, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# --- Imports and Path Setup ---
script_dir = Path(__file__).parent.resolve()
app_dir = script_dir / "naini_app"
if str(app_dir) not in sys.path: sys.path.insert(0, str(app_dir)); print(f"Added {app_dir} to sys.path")
try:
    from naini_app import config
    # Ensure main, rag_chain, rag_indexing have the modified functions
    from naini_app.main import process_pdf_to_markdown
    from naini_app.rag_chain import setup_rag_chain_components
    from naini_app.rag_indexing import build_and_save_faiss_index # Might not be called directly here, but ensure it's available if needed
except ImportError as e: print(f"FATAL: Error importing Naini modules: {e}"); sys.exit(1)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s', force=True)
log = logging.getLogger(__name__)

# --- Global State ---
MARKDOWN_OUTPUT_PATH: Optional[Path] = None # Path for the *default* MD
LLM_INSTANCE: Optional[HuggingFacePipeline] = None
RAW_PIPELINE: Optional[Any] = None
TOKENIZER_INSTANCE: Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]] = None
EMBEDDING_INSTANCE: Optional[HuggingFaceEmbeddings] = None
VECTOR_STORE: Optional[FAISS] = None # Holds the currently active vector store
RAG_CHAIN_PROMPT_GENERATOR: Optional[Runnable] = None # Holds the currently active prompt generator
RAG_CHAIN: Optional[Runnable] = None # Holds the FULL active RAG chain (PromptGen | Custom Streamer)
INITIALIZATION_ERROR: Optional[str] = None
ACTIVE_DOCUMENT_INFO: Dict[str, Any] = { # Track active document details
    "name": "None",
    "markdown_path": None,
    "index_name": None,
    "is_default": True # Initially, the default document is active
}
LAYOUT_PROCESSOR: Optional[Any] = None # NEW: Global for layout processor
LAYOUT_MODEL: Optional[Any] = None     # NEW: Global for layout model

# --- Helper Function ---
def calculate_pdf_hash(pdf_path: Path) -> str:
    """Calculates SHA256 hash of the PDF file content."""
    hasher = hashlib.sha256()
    try:
        with open(pdf_path, 'rb') as f:
            while True:
                chunk = f.read(8192) # Read in chunks
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        log.error(f"Error calculating hash for {pdf_path}: {e}", exc_info=True)
        return "error_calculating_hash"

# --- Setup Functions (Default Doc & Models) ---
def ensure_default_markdown():
    """Checks for default Markdown, processes if needed. Returns True on success."""
    global MARKDOWN_OUTPUT_PATH, INITIALIZATION_ERROR, ACTIVE_DOCUMENT_INFO
    target_md_path = config.DEFAULT_PDF_OUTPUT_MARKDOWN_PATH
    log.info(f"Step 1: Checking for default Markdown at {target_md_path}...")
    if target_md_path.is_file():
        log.info("Found existing default Markdown file.")
        MARKDOWN_OUTPUT_PATH = target_md_path
        ACTIVE_DOCUMENT_INFO = { # Initialize active doc to default
            "name": config.DEFAULT_PDF_PATH.name,
            "markdown_path": target_md_path,
            "index_name": config.FAISS_INDEX_NAME, # Default index name from config
            "is_default": True
        }
        log.info(f"Active document initialized to default: {config.DEFAULT_PDF_PATH.name}")
        return True
    else:
        log.warning(f"Default Markdown file not found at {target_md_path}. Starting processing from {config.DEFAULT_PDF_PATH}...")
        start_time = time.time()
        try:
            # Call the modified process_pdf_to_markdown, ensuring correct args
            # For default, output MD to its specific subdir, intermediates to persistent dirs
            result_path = process_pdf_to_markdown(
                pdf_path_str=str(config.DEFAULT_PDF_PATH),
                output_markdown_dir_str=str(config.DEFAULT_PDF_OUTPUT_MARKDOWN_PATH.parent), # Dir: .../output_markdown/default_processed/
                output_image_dir_str=str(config.DEFAULT_PERSISTENT_IMAGE_DIR),          # Persistent dir for default
                output_single_page_pdf_dir_str=str(config.DEFAULT_PERSISTENT_SINGLE_PDF_DIR), # Persistent dir for default
                layout_processor=LAYOUT_PROCESSOR, # <-- Pass global
                layout_model=LAYOUT_MODEL          # <-- Pass global
            )
            end_time = time.time()

            # Check if the expected file was created
            if result_path and result_path == config.DEFAULT_PDF_OUTPUT_MARKDOWN_PATH:
                 MARKDOWN_OUTPUT_PATH = result_path
                 ACTIVE_DOCUMENT_INFO = { # Initialize active doc to default
                     "name": config.DEFAULT_PDF_PATH.name,
                     "markdown_path": result_path,
                     "index_name": config.FAISS_INDEX_NAME,
                     "is_default": True
                 }
                 log.info(f"Processing successful ({end_time - start_time:.2f}s). Markdown saved to {result_path}")
                 log.info(f"Active document initialized to default: {config.DEFAULT_PDF_PATH.name}")
                 return True
            else:
                error_msg = f"Processing ran but expected output file is missing or incorrect: {config.DEFAULT_PDF_OUTPUT_MARKDOWN_PATH}"
                log.error(error_msg)
                INITIALIZATION_ERROR = error_msg
                return False
        except Exception as e:
            error_msg = f"Error during default PDF processing: {e}"
            log.error(error_msg, exc_info=True)
            INITIALIZATION_ERROR = error_msg
            return False

def load_models():
    """Loads models. Stores both LangChain wrapper and RAW pipeline."""
    # (Implementation unchanged from previous version)
    global EMBEDDING_INSTANCE, LLM_INSTANCE, RAW_PIPELINE, TOKENIZER_INSTANCE, INITIALIZATION_ERROR
    log.info("Step 2: Loading models (using default Hugging Face cache)...")
    try:
        # Embedding Model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        EMBEDDING_INSTANCE = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_ID, model_kwargs={'device': device})
        log.info(f"Embedding model loaded successfully on {device}.")

        # LLM & Tokenizer
        log.info(f"Loading LLM & Tokenizer (Model: {config.LLM_MODEL_ID})...")
        quant_config = None; compute_dtype = torch.float16
        llm_quantization = getattr(config, 'LLM_QUANTIZATION', 'none').lower()
        if llm_quantization == "4bit":
            log.info("Using 4-bit quantization (BitsAndBytes)")
            if torch.cuda.is_available():
                quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
                compute_dtype = torch.bfloat16; log.info("4-bit config applied.")
            else: log.warning("GPU not available for 4-bit. Switching to 'none'."); llm_quantization = "none"
        elif llm_quantization == "8bit":
             log.info("Using 8-bit quantization (BitsAndBytes)")
             if torch.cuda.is_available():
                 quant_config = BitsAndBytesConfig(load_in_8bit=True); log.info("8-bit config applied.")
             else: log.warning("GPU not available for 8-bit. Switching to 'none'."); llm_quantization = "none"
        else: log.info("No quantization specified.")

        device_map = getattr(config, 'DEVICE_MAP', 'auto') if torch.cuda.is_available() else None
        log.info(f"LLM device map set to: {device_map or 'CPU'}")

        log.info("Loading tokenizer...")
        TOKENIZER_INSTANCE = AutoTokenizer.from_pretrained(config.LLM_MODEL_ID, trust_remote_code=True, use_fast=False)
        stop_token_ids = []
        try:
            eos_token_id = TOKENIZER_INSTANCE.eos_token_id
            im_end_token_id = TOKENIZER_INSTANCE.convert_tokens_to_ids("<|im_end|>")
            if eos_token_id is not None: stop_token_ids.append(eos_token_id)
            if im_end_token_id is not None and im_end_token_id not in stop_token_ids: stop_token_ids.append(im_end_token_id)
            log.info(f"Stop token IDs identified (for eos_token_id): {stop_token_ids}")
            if not stop_token_ids: log.warning("Could not identify standard stop tokens (EOS, <|im_end|>)!")
        except Exception as e_tok:
            log.warning(f"Could not get specific stop tokens: {e_tok}")
            if TOKENIZER_INSTANCE.eos_token_id is not None: stop_token_ids = [TOKENIZER_INSTANCE.eos_token_id]
            else: log.error("CRITICAL: Tokenizer has no EOS token ID!"); stop_token_ids = []
        if TOKENIZER_INSTANCE.pad_token is None:
            if TOKENIZER_INSTANCE.eos_token is not None: TOKENIZER_INSTANCE.pad_token = TOKENIZER_INSTANCE.eos_token; TOKENIZER_INSTANCE.pad_token_id = TOKENIZER_INSTANCE.eos_token_id; log.info("Tokenizer pad_token set to eos_token.")
            else: log.warning("Tokenizer has no pad_token and no eos_token to fall back on!")
        TOKENIZER_INSTANCE.padding_side = "right"
        log.info("Tokenizer loaded.")

        log.info("Loading base LLM (using default cache)...")
        model = AutoModelForCausalLM.from_pretrained(config.LLM_MODEL_ID, quantization_config=quant_config, device_map=device_map, trust_remote_code=True, torch_dtype=compute_dtype)
        model.eval(); log.info("Base LLM loaded.")

        log.info("Creating text-generation pipeline...")
        max_new = getattr(config, 'MAX_NEW_TOKENS', 512); temp = getattr(config, 'TEMPERATURE', 0.6); top_p_val = getattr(config, 'TOP_P', 0.9); do_sample_flag = getattr(config, 'DO_SAMPLE', True)
        RAW_PIPELINE = pipeline(
            "text-generation", model=model, tokenizer=TOKENIZER_INSTANCE,
            max_new_tokens=max_new, temperature=temp, top_p=top_p_val, do_sample=do_sample_flag,
            repetition_penalty=1.1,
            eos_token_id=TOKENIZER_INSTANCE.eos_token_id, # Use list if needed
            pad_token_id=TOKENIZER_INSTANCE.pad_token_id,
        )
        if not RAW_PIPELINE: raise ValueError("Failed to create raw transformers pipeline.")
        log.info("Raw transformers pipeline created successfully.")
        LLM_INSTANCE = HuggingFacePipeline(pipeline=RAW_PIPELINE)
        log.info("LangChain HuggingFacePipeline wrapper created.")
        return True
    except Exception as e: error_msg = f"Fatal error during LLM/Tokenizer loading: {e}"; log.error(error_msg, exc_info=True); INITIALIZATION_ERROR = error_msg; return False

# --- Custom Streaming Function ---
def stream_llm_response(prompt_result: Any) -> Generator[str, None, None]:
    """Streams response from the RAW pipeline."""
    # (Implementation unchanged from previous version)
    global RAW_PIPELINE, TOKENIZER_INSTANCE
    if not RAW_PIPELINE or not TOKENIZER_INSTANCE:
        log.error("Cannot stream: Raw pipeline or tokenizer not initialized.")
        yield "[ERROR: Streaming components not ready]"
        return

    try:
        prompt_string = prompt_result.to_string()
    except AttributeError:
        log.error(f"Input to stream_llm_response has no .to_string() method. Type: {type(prompt_result)}", exc_info=True)
        yield f"[ERROR: Invalid input type '{type(prompt_result)}' for streaming]"
        return

    log.debug(f"--- Starting stream_llm_response --- Prompt String (start):\n{prompt_string[:500]}...")
    streamer = TextIteratorStreamer(TOKENIZER_INSTANCE, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(text_inputs=prompt_string, streamer=streamer)
    thread = Thread(target=RAW_PIPELINE, kwargs=generation_kwargs)
    thread.start()
    log.debug("Pipeline thread started.")
    try:
        for new_text in streamer:
            yield new_text
    except Exception as e:
        log.error(f"Error occurred during streaming loop: {e}", exc_info=True)
        yield f"[ERROR during streaming: {e}]"
    finally:
        thread.join()
        log.debug("Pipeline thread finished.")

# --- Chat Interaction Functions ---
def add_text_to_history(history: List[Tuple[Optional[str], Optional[str]]], text: str) -> Tuple[List[Tuple[Optional[str], Optional[str]]], gr.Textbox]:
    """Adds user text to history, clears input."""
    # (Implementation unchanged)
    if not text: return history, gr.update()
    history.append((text, None))
    return history, gr.update(value="", interactive=False)

def generate_bot_response(history: List[Tuple[Optional[str], Optional[str]]]) -> Generator[List[Tuple[Optional[str], Optional[str]]], None, None]:
    """Runs the RAG chain for the ACTIVE document and streams response."""
    # (Implementation updated in previous step - using ACTIVE_DOCUMENT_INFO)
    global RAG_CHAIN, INITIALIZATION_ERROR, ACTIVE_DOCUMENT_INFO

    if not history or history[-1][1] is not None:
        yield history; return

    if INITIALIZATION_ERROR and ACTIVE_DOCUMENT_INFO.get("is_default", True):
        history[-1] = (history[-1][0], f"Initialisation Error: {INITIALIZATION_ERROR}")
        yield history; return
    if not RAG_CHAIN:
        error_msg = "Error: The RAG Chain for the active document is not ready."
        log.error(error_msg)
        history[-1] = (history[-1][0], error_msg)
        yield history; return

    user_query = history[-1][0]
    if not user_query:
        history[-1] = (history[-1][0], "Error: Empty query received.")
        yield history; return

    active_doc_name = ACTIVE_DOCUMENT_INFO.get('name', 'Unknown')
    log.info(f"--- Streaming RAG Chain for query about '{active_doc_name}': '{user_query}' ---")

    history[-1] = (user_query, "")
    yield history

    chat_history_messages: List[BaseMessage] = []
    for user_msg, ai_msg in history[:-1]:
        if user_msg: chat_history_messages.append(HumanMessage(content=user_msg))
        if ai_msg: chat_history_messages.append(AIMessage(content=ai_msg))

    full_response = ""
    stream_started = False
    try:
        chain_input = {"question": user_query, "chat_history": chat_history_messages}
        log.info(f"Streaming RAG chain for '{active_doc_name}' (PromptGen | Custom Streamer)...")
        start_stream_time = time.time()

        for token in RAG_CHAIN.stream(chain_input, config=RunnableConfig(callbacks=[])):
            if isinstance(token, str):
                full_response += token
                history[-1] = (user_query, full_response)
                stream_started = True
                yield history
            else:
                log.warning(f"Received non-string token from custom stream: type={type(token)}, value={token}")

        end_stream_time = time.time()
        if not stream_started:
            log.warning(f"RAG Chain stream for '{active_doc_name}' finished ({end_stream_time - start_stream_time:.2f}s) but produced no string tokens.")
            if not history[-1][1]:
                 history[-1] = (user_query, "Sorry, I couldn't generate a response (empty stream).")
            yield history
        else:
             log.info(f"RAG Chain stream for '{active_doc_name}' finished in {end_stream_time - start_stream_time:.2f}s. Final length: {len(full_response)}.")
             if history[-1][1] != full_response:
                 history[-1] = (user_query, full_response)
                 yield history

    except Exception as e:
        log.error(f"Error during RAG chain streaming for '{active_doc_name}': {e}", exc_info=True)
        error_context = f"\n\n[DEBUG: Partial Response before error:\n{full_response[:500]}...]" if full_response else ""
        history[-1] = (user_query, f"Sorry, an error occurred while generating the response for '{active_doc_name}': {e}{error_context}")
        yield history

# --- Pre-Launch Setup ---
setup_successful = False
if __name__ == "__main__":
    log_level = logging.INFO
    logging.getLogger().setLevel(log_level)
    actual_level_num = logging.getLogger().getEffectiveLevel()
    actual_level_name = logging.getLevelName(actual_level_num)
    log.info(f"--- Logging Level Set To: {actual_level_name} ---")

    log.info("==========================================")
    log.info("  Naini Initialization Sequence Started   ")
    log.info("==========================================")
    overall_start_time = time.time()

    # Initialize with Default Document
    if ensure_default_markdown(): # This now also sets initial ACTIVE_DOCUMENT_INFO
        log.info(f"Step 1 OK: Default Markdown path confirmed: {MARKDOWN_OUTPUT_PATH}")
        if load_models():
            if not RAW_PIPELINE:
                 log.error("FATAL: Raw pipeline object was not created during model loading.")
                 INITIALIZATION_ERROR = INITIALIZATION_ERROR or "Raw pipeline missing after load_models."
                 # Exit or proceed to UI in error state based on desired behavior
            else:
                log.info(f"Step 2 OK: Core LLM/Embedding models loaded.")
                # --- <<< ADD LAYOUT MODEL LOADING HERE >>> ---
            log.info("Step 2.5: Loading Layout Detection Model...")
            layout_load_start_time = time.time()
            try:
                # Call the loader from layout_detector.py
                LAYOUT_PROCESSOR, LAYOUT_MODEL = load_layout_model_and_processor()
                layout_load_end_time = time.time()
                if LAYOUT_PROCESSOR and LAYOUT_MODEL:
                    log.info(f"Step 2.5 OK: Layout model loaded ({layout_load_end_time - layout_load_start_time:.2f}s).")
                    # Proceed only if layout model also loaded successfully
                    log.info("Step 3: Setting up RAG chain components for DEFAULT document...")
                    rag_setup_start_time = time.time()
                    # Call setup_rag_chain_components for the default document
                    vector_store_instance, rag_chain_prompt_gen_instance, rag_error = setup_rag_chain_components(
                        markdown_path=config.DEFAULT_PDF_OUTPUT_MARKDOWN_PATH,
                        embedding_model=EMBEDDING_INSTANCE,
                        index_folder_path=config.RAG_CACHE_DIR,
                        index_name=config.FAISS_INDEX_NAME
                    )
                    rag_setup_end_time = time.time()
                    log.info(f"Default RAG setup attempt finished in {rag_setup_end_time - rag_setup_start_time:.2f}s.")

                    if rag_error is None and vector_store_instance is not None and rag_chain_prompt_gen_instance is not None:
                        VECTOR_STORE = vector_store_instance
                        RAG_CHAIN_PROMPT_GENERATOR = rag_chain_prompt_gen_instance
                        log.info(f"Step 3 OK: Default RAG components ready.")

                        log.info("Step 4: Constructing Full RAG Chain for DEFAULT document...")
                        try:
                            RAG_CHAIN = RAG_CHAIN_PROMPT_GENERATOR | RunnableLambda(stream_llm_response)
                            log.info("Step 4 OK: Default Full RAG chain constructed.")
                            setup_successful = True # <<< MARK SETUP SUCCESSFUL HERE >>>
                        except Exception as e:
                            INITIALIZATION_ERROR = f"Failed to construct default RAG chain: {e}"
                            log.error(f"FATAL: Default RAG chain construction failed (Step 4). Error: {INITIALIZATION_ERROR}", exc_info=True)
                    else: # Default RAG setup failed
                        if vector_store_instance: VECTOR_STORE = vector_store_instance
                        INITIALIZATION_ERROR = rag_error or "Unknown default RAG setup error (Step 3)."
                        log.error(f"FATAL: Default RAG component setup failed (Step 3). Error: {INITIALIZATION_ERROR}")

                else: # Layout model loading failed
                    INITIALIZATION_ERROR = "Failed to load Layout Detection model."
                    log.error(f"FATAL: {INITIALIZATION_ERROR} (Step 2.5)")
            except Exception as layout_e:
                INITIALIZATION_ERROR = f"Exception during layout model loading: {layout_e}"
                log.error(f"FATAL: {INITIALIZATION_ERROR} (Step 2.5)", exc_info=True)
            # --- <<< END OF LAYOUT MODEL LOADING >>> ---

                log.info("Step 3: Setting up RAG chain components for DEFAULT document...")
                rag_setup_start_time = time.time()
                # Call setup_rag_chain_components for the default document using paths from config
                vector_store_instance, rag_chain_prompt_gen_instance, rag_error = setup_rag_chain_components(
                    markdown_path=config.DEFAULT_PDF_OUTPUT_MARKDOWN_PATH, # Default MD path
                    embedding_model=EMBEDDING_INSTANCE,
                    index_folder_path=config.RAG_CACHE_DIR, # Default cache dir
                    index_name=config.FAISS_INDEX_NAME       # Default index name
                )
                rag_setup_end_time = time.time()
                log.info(f"Default RAG setup attempt finished in {rag_setup_end_time - rag_setup_start_time:.2f}s.")

                if rag_error is None and vector_store_instance is not None and rag_chain_prompt_gen_instance is not None:
                    VECTOR_STORE = vector_store_instance # Set global for default
                    RAG_CHAIN_PROMPT_GENERATOR = rag_chain_prompt_gen_instance # Set global for default
                    log.info(f"Step 3 OK: Default RAG components ready (Index: {config.FAISS_INDEX_NAME}).")

                    log.info("Step 4: Constructing Full RAG Chain for DEFAULT document...")
                    try:
                        RAG_CHAIN = RAG_CHAIN_PROMPT_GENERATOR | RunnableLambda(stream_llm_response)
                        log.info("Step 4 OK: Default Full RAG chain constructed.")
                        setup_successful = True
                    except Exception as e:
                        INITIALIZATION_ERROR = f"Failed to construct default RAG chain: {e}"
                        log.error(f"FATAL: Default RAG chain construction failed (Step 4). Error: {INITIALIZATION_ERROR}", exc_info=True)
                else:
                    if vector_store_instance: VECTOR_STORE = vector_store_instance # Still store if loaded but chain failed
                    INITIALIZATION_ERROR = rag_error or "Unknown default RAG setup error (Step 3)."
                    log.error(f"FATAL: Default RAG component setup failed (Step 3). Error: {INITIALIZATION_ERROR}")
        else:
            log.error("FATAL: Model loading failed (Step 2).")
    else:
        log.error("FATAL: Default Markdown processing/check failed (Step 1).")

    if setup_successful:
        overall_end_time = time.time()
        log.info(f"Naini Initialization Successful ({overall_end_time - overall_start_time:.2f}s)")
    else:
        if not INITIALIZATION_ERROR: INITIALIZATION_ERROR = "Unknown initialization failure."
        log.error(f"Initialization failed. Final Error Recorded: {INITIALIZATION_ERROR}")

    log.info("===========================================")

# --- Build Gradio Interface (V1.3 - Uploads & UI Refresh) ---
initial_chatbot_message = []
default_doc_display_name = config.DEFAULT_PDF_PATH.name if hasattr(config, 'DEFAULT_PDF_PATH') and config.DEFAULT_PDF_PATH else "the default document"
if setup_successful and ACTIVE_DOCUMENT_INFO.get("is_default", False):
    initial_chatbot_message = [(None, f"Hello! I'm ready to answer questions about the default document: **'{default_doc_display_name}'**. You can also upload your own PDF below.")]
elif not setup_successful and MARKDOWN_OUTPUT_PATH: # Default MD exists, but models/RAG failed
     initial_chatbot_message = [(None, f"ERROR: Naini processed the default document but failed to load models or RAG chain. Check logs.\nDetails: {INITIALIZATION_ERROR}")]
else: # Default MD processing failed or other init error
     initial_chatbot_message = [(None, f"ERROR: Naini failed during initialization. Check logs.\nDetails: {INITIALIZATION_ERROR}")]

app_title = getattr(config, 'APP_TITLE', 'Naini Document Assistant')
creator_name = getattr(config, 'CREATOR_NAME', "Wizard Sai")
naini_image_path = getattr(config, 'NAINI_PIC_PATH', None)
naini_avatar_path = str(naini_image_path) if naini_image_path and Path(naini_image_path).exists() else None

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="orange"), title=app_title) as demo:

    # --- Top Row: Title & Creator ---
    with gr.Row():
        #  gr.Markdown(f"# {app_title}", scale=3)
        gr.Markdown(f"# {app_title}")
        #gr.Markdown(f"*{creator_name}*", scale=1, elem_classes=["text-right"]) # Align right if possible via CSS
        gr.Markdown(f"by: *{creator_name}*", elem_classes=["text-right"])
    # --- Main Content Row ---
    with gr.Row():
        # --- Left Column: Status, Default Doc, Upload ---
        with gr.Column(scale=2, min_width=300): # Adjusted scale slightly
            if naini_image_path and Path(naini_image_path).exists():
                gr.Image(str(naini_image_path), width=150, height=150, label="Naini", scale=0, container=False, elem_classes=["center-img"]) # Centering might need CSS
            else:
                gr.Markdown("*(Naini image not found)*")

            with gr.Accordion("System Status", open=False): # Initially closed
                status_ready = getattr(config, 'STATUS_READY', "Ready"); status_error = getattr(config, 'STATUS_ERROR', "Error")
                status_text = status_ready if setup_successful else status_error
                status_details = "Default document loaded and ready." if setup_successful else f"Initialization failed. {INITIALIZATION_ERROR or 'Check logs.'}"
                gr.Textbox(label="Init Status", value=status_text, interactive=False)
                gr.Textbox(label="Init Details", value=status_details, lines=2, interactive=False)

            with gr.Accordion("Default Document Info", open=False): # Initially closed
                gr.Textbox(label="Name", value=default_doc_display_name, interactive=False)
                gr.File(label="Download Default PDF", value=str(config.DEFAULT_PDF_PATH), interactive=False) # Download button

            with gr.Accordion("Upload & Process PDF", open=True): # Initially open
                pdf_upload_component = gr.File(
                    label="Upload PDF File",
                    file_types=[".pdf"],
                    type="filepath",
                    interactive=setup_successful # Only enable if init worked
                )
                upload_status_textbox = gr.Textbox(
                    label="Upload/Processing Status",
                    value="Waiting for upload..." if setup_successful else "Unavailable (Init Failed)",
                    interactive=False,
                    lines=2 # Allow slightly more text
                )
                process_upload_button = gr.Button(
                    "Process Uploaded PDF",
                    variant="secondary",
                    interactive=False # Disabled until file uploaded
                )

        # --- Right Column: Chat Interface ---
        with gr.Column(scale=5): # Adjusted scale slightly
            gr.Markdown("## Document Interaction")
            active_doc_display = gr.Textbox(
                label="Currently Active Document",
                # Use ACTIVE_DOCUMENT_INFO which is set during init and upload processing
                value=f"{'Default: ' if ACTIVE_DOCUMENT_INFO.get('is_default') else ''}{ACTIVE_DOCUMENT_INFO.get('name', 'None')}" if setup_successful else "None (Initialization Failed)",
                interactive=False
            )

            chatbot_ui = gr.Chatbot(
                label="Chat with Naini",
                value=initial_chatbot_message,
                height=650, # Slightly increased height
                visible=True,
                avatar_images=(None, naini_avatar_path),
                render_markdown=True,
                show_copy_button=True,
                bubble_full_width=False
            )
            with gr.Row():
                 chat_input_box = gr.Textbox(
                     label="Ask a question", scale=4, show_label=False,
                     placeholder="Ask about the active document..." if setup_successful else "Chat disabled.",
                     container=False, visible=True, interactive=setup_successful
                 )
                 submit_button = gr.Button(
                     "Ask", scale=1, variant="primary", visible=True, interactive=setup_successful
                 )

    # --- Define Gradio Actions ---

    # Action 1: Enable Process button when a file is uploaded
    def handle_file_upload(filepath: Optional[str]) -> Tuple[gr.Textbox, gr.Button]:
        """Updates UI when file is uploaded."""
        if filepath and Path(filepath).exists():
            log.info(f"File uploaded: {filepath}")
            return gr.update(value=f"File '{Path(filepath).name}' ready. Click Process."), gr.update(interactive=True)
        else:
            log.warning("File upload event received, but path is invalid or cleared.")
            # Reset status only if a file isn't actively selected
            return gr.update(value="Waiting for upload..."), gr.update(interactive=False)

    if setup_successful: # Only bind upload action if init worked
        pdf_upload_component.upload(
            fn=handle_file_upload,
            inputs=[pdf_upload_component],
            outputs=[upload_status_textbox, process_upload_button]
        )
        # Also handle clearing the file upload
        pdf_upload_component.clear(
             fn=lambda: (gr.update(value="Waiting for upload..."), gr.update(interactive=False)),
             inputs=None,
             outputs=[upload_status_textbox, process_upload_button]
        )

    # Action 2: Process the uploaded PDF (Hashing, Caching, Processing, Indexing, RAG Update)
    # Action 2: Process the uploaded PDF (Persistent Storage Version)
    def handle_process_upload_click(
        uploaded_pdf_path_obj: Optional[gr.utils.NamedString],
        ) -> Generator[Dict[gr.component, Any], None, None]:
        """
        Handles processing of uploaded PDF, storing PDF & MD persistently.
        1. Calculates PDF hash.
        2. Checks persistent storage for existing Markdown based on hash.
        3. If cache miss:
           - Copies original PDF to persistent location.
           - Processes PDF (using temp intermediates, temp initial MD output).
           - Copies generated MD to persistent location.
        4. Loads/Builds FAISS index (using TEMPORARY index cache).
        5. Updates global RAG_CHAIN.
        6. Updates UI elements and clears chat history.
        """
        global RAG_CHAIN, RAG_CHAIN_PROMPT_GENERATOR, VECTOR_STORE, ACTIVE_DOCUMENT_INFO, EMBEDDING_INSTANCE, LLM_INSTANCE, RAW_PIPELINE, TOKENIZER_INSTANCE, LAYOUT_PROCESSOR, LAYOUT_MODEL # Added Layout model globals

        # --- Initial UI Update & Input Validation ---
        yield {
            upload_status_textbox: gr.update(value="Starting processing..."),
            active_doc_display: gr.update(value="Processing Upload..."),
            chatbot_ui: gr.update(value=[(None, "Processing request received...")]), # Clear history
            chat_input_box: gr.update(interactive=False, placeholder="Processing... Please wait."),
            process_upload_button: gr.update(interactive=False) # Disable
        }

        if not uploaded_pdf_path_obj or not uploaded_pdf_path_obj.name:
            log.error("Process button clicked, but no valid file path received.")
            yield { # Return error state
                upload_status_textbox: gr.update(value="Error: No file provided."),
                active_doc_display: gr.update(value="Error"),
                chat_input_box: gr.update(interactive=False, placeholder="Error: No file provided."),
                process_upload_button: gr.update(interactive=True),
                chatbot_ui: gr.update(value=[])
            }
            return

        uploaded_pdf_path = Path(uploaded_pdf_path_obj.name) # Temp path from Gradio
        if not uploaded_pdf_path.exists():
            log.error(f"Uploaded file path does not exist: {uploaded_pdf_path}")
            yield { # Return error state
                upload_status_textbox: gr.update(value=f"Error: File not found at {uploaded_pdf_path}"),
                active_doc_display: gr.update(value="Error"),
                chat_input_box: gr.update(interactive=False, placeholder="Error: File not found."),
                process_upload_button: gr.update(interactive=True),
                chatbot_ui: gr.update(value=[])
            }
            return

        pdf_name = uploaded_pdf_path.name
        log.info(f"Processing request for: {pdf_name} ({uploaded_pdf_path})")

        # Ensure necessary base components are loaded
        if not all([EMBEDDING_INSTANCE, LLM_INSTANCE, RAW_PIPELINE, TOKENIZER_INSTANCE, LAYOUT_PROCESSOR, LAYOUT_MODEL]):
             error_msg = "Core models (Embed/LLM/Layout) not loaded. Cannot process upload."
             log.error(error_msg)
             yield { # Return error state
                 upload_status_textbox: gr.update(value=f"Error: {error_msg}"),
                 active_doc_display: gr.update(value="System Error"),
                 chat_input_box: gr.update(interactive=False, placeholder=error_msg),
                 process_upload_button: gr.update(interactive=True),
                 chatbot_ui: gr.update(value=[])
             }
             return

        # --- 1. Calculate Hash ---
        yield {upload_status_textbox: gr.update(value=f"Calculating hash for '{pdf_name}'...")}
        pdf_hash = calculate_pdf_hash(uploaded_pdf_path)
        if pdf_hash == "error_calculating_hash":
             yield { # Return error state
                 upload_status_textbox: gr.update(value=f"Error calculating hash for '{pdf_name}'."),
                 active_doc_display: gr.update(value="Error"),
                 chat_input_box: gr.update(interactive=False, placeholder="Hashing error."),
                 process_upload_button: gr.update(interactive=True),
                 chatbot_ui: gr.update(value=[])
             }
             return
        log.info(f"PDF Hash ({pdf_name}): {pdf_hash}")

        # --- 2. Define Persistent Paths & Check Persistent Cache ---
        persistent_doc_dir = config.PERSISTENT_USER_UPLOAD_DIR / pdf_hash
        persistent_pdf_copy_path = persistent_doc_dir / f"original_{pdf_name}" # Store original PDF
        persistent_md_path = persistent_doc_dir / "processed_markdown.md"      # Standard MD name

        # --- Define TEMPORARY paths for this run (if needed for processing) ---
        # Temporary MD filename (used if process_pdf_to_markdown is called)
        safe_pdf_stem = re.sub(r'[^\w\-]+', '_', uploaded_pdf_path.stem)
        temp_markdown_filename = f"{pdf_hash}_{safe_pdf_stem}_output.md"
        temp_md_output_path = config.TEMP_MARKDOWN_DIR / temp_markdown_filename
        # Temporary intermediate dirs (used if process_pdf_to_markdown is called)
        run_temp_intermediate_dir = config.TEMP_UPLOADS_DIR / pdf_hash
        temp_image_dir = run_temp_intermediate_dir / "images"
        temp_single_page_pdf_dir = run_temp_intermediate_dir / "single_pdfs"
        # Temporary index details (always used, index not persistent yet)
        temp_index_name = f"idx_{pdf_hash}_{safe_pdf_stem}"
        temp_index_folder_path = config.TEMP_INDEX_DIR

        final_md_path_for_rag: Optional[Path] = None
        is_cache_hit = False

        yield {upload_status_textbox: gr.update(value=f"Checking persistent storage for '{pdf_name}'...")}
        if persistent_md_path.is_file():
            # --- CACHE HIT ---
            log.info(f"Persistent Cache HIT: Found existing Markdown at {persistent_md_path}")
            final_md_path_for_rag = persistent_md_path
            is_cache_hit = True
            # Also check if original PDF copy exists, copy if not (optional integrity check)
            if not persistent_pdf_copy_path.exists():
                 log.warning(f"Markdown cache hit, but original PDF copy missing at {persistent_pdf_copy_path}. Copying now.")
                 try:
                     persistent_doc_dir.mkdir(parents=True, exist_ok=True)
                     shutil.copy2(uploaded_pdf_path, persistent_pdf_copy_path)
                 except Exception as copy_err:
                      log.error(f"Error copying original PDF to persistent storage (during cache hit check): {copy_err}")
                      # Decide how critical this is - maybe still proceed?
        else:
            # --- CACHE MISS ---
            log.info(f"Persistent Cache MISS: No Markdown found at {persistent_md_path}. Processing PDF.")
            yield {upload_status_textbox: gr.update(value=f"Processing '{pdf_name}' (this may take time)...")}

            # 3a. Create persistent dir & Copy original PDF
            try:
                persistent_doc_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(uploaded_pdf_path, persistent_pdf_copy_path)
                log.info(f"Copied original PDF to persistent storage: {persistent_pdf_copy_path}")
            except Exception as copy_e:
                log.error(f"FATAL: Failed to copy original PDF to persistent storage {persistent_pdf_copy_path}: {copy_e}")
                yield { # Return error state - Cannot proceed without PDF copy
                    upload_status_textbox: gr.update(value=f"Error copying PDF: {copy_e}"),
                    active_doc_display: gr.update(value="Storage Error"),
                    chat_input_box: gr.update(interactive=False, placeholder="Storage error."),
                    process_upload_button: gr.update(interactive=True),
                    chatbot_ui: gr.update(value=[])
                }
                return

            # 3b. Process PDF (using temp intermediates and temp initial MD output path)
            start_process_time = time.time()
            processed_md_temp_path: Optional[Path] = None
            try:
                 # Ensure necessary TEMP directories exist for this run
                 temp_image_dir.mkdir(parents=True, exist_ok=True)
                 temp_single_page_pdf_dir.mkdir(parents=True, exist_ok=True)
                 config.TEMP_MARKDOWN_DIR.mkdir(parents=True, exist_ok=True) # Dir for initial MD output

                 processed_md_temp_path = process_pdf_to_markdown(
                     pdf_path_str=str(uploaded_pdf_path),       # Use temp path for processing input
                     output_markdown_dir_str=str(config.TEMP_MARKDOWN_DIR), # Initial MD output to temp dir
                     output_image_dir_str=str(temp_image_dir),            # Temp intermediates
                     output_single_page_pdf_dir_str=str(temp_single_page_pdf_dir), # Temp intermediates
                     layout_processor=LAYOUT_PROCESSOR,
                     layout_model=LAYOUT_MODEL
                 )
                 end_process_time = time.time()

                 if processed_md_temp_path and processed_md_temp_path.is_file():
                     log.info(f"PDF processing successful ({end_process_time - start_process_time:.2f}s). Temp MD: {processed_md_temp_path}")

                     # 3c. Copy generated temporary MD to final persistent location
                     try:
                         shutil.copy2(processed_md_temp_path, persistent_md_path)
                         log.info(f"Copied processed Markdown to persistent storage: {persistent_md_path}")
                         final_md_path_for_rag = persistent_md_path # Use the persistent path for RAG

                         # Optional: Clean up temporary MD file after successful copy
                         try: processed_md_temp_path.unlink(); log.debug(f"Removed temp MD file: {processed_md_temp_path}")
                         except OSError as del_e: log.warning(f"Could not remove temp MD file {processed_md_temp_path}: {del_e}")

                     except Exception as copy_md_e:
                          log.error(f"FATAL: Failed to copy processed Markdown from {processed_md_temp_path} to {persistent_md_path}: {copy_md_e}")
                          yield { # Return error state - MD processed but couldn't store persistently
                              upload_status_textbox: gr.update(value=f"Error saving processed file: {copy_md_e}"),
                              active_doc_display: gr.update(value="Storage Error"),
                              chat_input_box: gr.update(interactive=False, placeholder="Storage error."),
                              process_upload_button: gr.update(interactive=True),
                              chatbot_ui: gr.update(value=[])
                          }
                          return # Cannot proceed if MD cannot be saved persistently

                 else: # process_pdf_to_markdown failed
                     error_msg = f"PDF processing failed for '{pdf_name}'. Check worker logs."
                     log.error(error_msg)
                     yield { # Return error state
                         upload_status_textbox: gr.update(value=error_msg),
                         active_doc_display: gr.update(value="Processing Error"),
                         chatbot_ui: gr.update(value=[(None, f"Error: Failed to process '{pdf_name}'.")]),
                         chat_input_box: gr.update(interactive=False, placeholder="Processing failed."),
                         process_upload_button: gr.update(interactive=True)
                     }
                     return

            except Exception as proc_e:
                 log.error(f"Exception during PDF processing stage for {pdf_name}: {proc_e}", exc_info=True)
                 yield { # Return error state
                     upload_status_textbox: gr.update(value=f"Error during processing: {proc_e}"),
                     active_doc_display: gr.update(value="Processing Error"),
                     chatbot_ui: gr.update(value=[(None, f"Error: Exception during processing of '{pdf_name}'.")]),
                     chat_input_box: gr.update(interactive=False, placeholder="Processing exception."),
                     process_upload_button: gr.update(interactive=True)
                 }
                 return
            finally:
                 # Optional: Cleanup intermediate image/single-page PDF directory for this run
                 try: shutil.rmtree(run_temp_intermediate_dir); log.info(f"Cleaned up temp intermediates: {run_temp_intermediate_dir}")
                 except Exception as clean_e: log.warning(f"Could not clean up intermediates dir {run_temp_intermediate_dir}: {clean_e}")

        # --- 4. Setup RAG Components (Using Persistent MD Path, Temporary Index) ---
        if final_md_path_for_rag and final_md_path_for_rag.is_file():
            yield {upload_status_textbox: gr.update(value=f"Loading/Indexing '{pdf_name}'...")}
            start_rag_setup_time = time.time()
            try:
                # Setup RAG using the final MD path (which is persistent)
                # Index is still temporary for now
                vector_store_instance, rag_chain_prompt_gen_instance, rag_error = setup_rag_chain_components(
                    markdown_path=final_md_path_for_rag,    # Use the determined MD path
                    embedding_model=EMBEDDING_INSTANCE,
                    index_folder_path=temp_index_folder_path, # Temp index folder
                    index_name=temp_index_name                # Temp index name
                )
                end_rag_setup_time = time.time()
                log.info(f"RAG setup for '{pdf_name}' completed in {end_rag_setup_time - start_rag_setup_time:.2f}s.")

                if rag_error is None and vector_store_instance is not None and rag_chain_prompt_gen_instance is not None:
                    # --- 5. Update Global RAG Chain & Active Document ---
                    log.info(f"Successfully set up RAG components for '{pdf_name}'. Updating global chain.")
                    VECTOR_STORE = vector_store_instance
                    RAG_CHAIN_PROMPT_GENERATOR = rag_chain_prompt_gen_instance
                    RAG_CHAIN = RAG_CHAIN_PROMPT_GENERATOR | RunnableLambda(stream_llm_response)
                    log.info("Global RAG_CHAIN updated.")

                    ACTIVE_DOCUMENT_INFO = {
                        "name": pdf_name,
                        "markdown_path": final_md_path_for_rag, # Store persistent path
                        "index_name": temp_index_name,          # Store temp index name
                        "is_default": False
                    }
                    log.info(f"Active document set to: {pdf_name} (using {final_md_path_for_rag})")

                    # --- 6. Final Success UI Update ---
                    cache_msg = "(from persistent storage)" if is_cache_hit else "(processed)"
                    success_msg = f"Ready! Ask questions about '{pdf_name}' {cache_msg}."
                    yield {
                        upload_status_textbox: gr.update(value=success_msg),
                        active_doc_display: gr.update(value=f"Active: {pdf_name}"),
                        chatbot_ui: gr.update(value=[(None, f"Hello! I have processed '{pdf_name}'. Ask me anything about it.")]), # Welcome msg
                        chat_input_box: gr.update(interactive=True, placeholder=f"Ask about '{pdf_name}'..."),
                        process_upload_button: gr.update(interactive=True) # Re-enable
                    }
                    log.info(f"Processing and loading complete for {pdf_name}.")

                else: # RAG setup failed
                    error_msg = f"Failed to set up RAG components for '{pdf_name}': {rag_error or 'Unknown error'}"
                    log.error(error_msg)
                    yield { # Return error state
                        upload_status_textbox: gr.update(value=error_msg),
                        active_doc_display: gr.update(value="RAG Setup Error"),
                        chatbot_ui: gr.update(value=[(None, f"Error: Failed RAG setup for '{pdf_name}'.")]),
                        chat_input_box: gr.update(interactive=False, placeholder="RAG setup failed."),
                        process_upload_button: gr.update(interactive=True)
                    }
            except Exception as rag_e:
                log.error(f"Exception during RAG setup for {pdf_name}: {rag_e}", exc_info=True)
                yield { # Return error state
                    upload_status_textbox: gr.update(value=f"Error during RAG setup: {rag_e}"),
                    active_doc_display: gr.update(value="RAG Setup Error"),
                    chatbot_ui: gr.update(value=[(None, f"Error: Exception during RAG setup for '{pdf_name}'.")]),
                    chat_input_box: gr.update(interactive=False, placeholder="RAG setup exception."),
                    process_upload_button: gr.update(interactive=True)
                 }
        else:
            # MD path is None or file doesn't exist (should have been caught earlier)
            log.error("Process/Cache check finished, but final_md_path_for_rag is invalid. Cannot proceed.")
            yield { # Return error state
                 upload_status_textbox: gr.update(value="Internal Error: Markdown path missing."),
                 active_doc_display: gr.update(value="Internal Error"),
                 chat_input_box: gr.update(interactive=False, placeholder="Internal error."),
                 process_upload_button: gr.update(interactive=True)
             }
    # --- End of handle_process_upload_click function ---


    # <<< --- Attach handle_process_upload_click to the button HERE --- >>>
    if setup_successful: # Only bind process button if init worked
        process_upload_button.click(
            fn=handle_process_upload_click,
            inputs=[pdf_upload_component], # Only need the file component path object
            # Use dictionary output format to target specific components by variable name
            outputs=[
                upload_status_textbox,
                active_doc_display,
                chatbot_ui,
                chat_input_box,
                process_upload_button,
            ]
        )
    # <<< --- End of process_upload_button.click() attachment --- >>>


    # Action 3: Chat interaction
    if setup_successful: # Only enable chat actions if initial setup worked
        submit_components = [chatbot_ui, chat_input_box]
        submit_action = (
            submit_button.click(
                fn=add_text_to_history, inputs=[chatbot_ui, chat_input_box], outputs=submit_components, queue=False
            ).then(
                fn=generate_bot_response, inputs=[chatbot_ui], outputs=[chatbot_ui], api_name=False # generate_bot_response uses the ACTIVE_DOCUMENT_INFO implicitly
            ).then(
                lambda: gr.update(interactive=True), inputs=None, outputs=[chat_input_box], api_name=False # Re-enable input
            )
        )
        enter_action = (
            chat_input_box.submit(
                fn=add_text_to_history, inputs=[chatbot_ui, chat_input_box], outputs=submit_components, queue=False
            ).then(
                fn=generate_bot_response, inputs=[chatbot_ui], outputs=[chatbot_ui], api_name=False
            ).then(
                lambda: gr.update(interactive=True), inputs=None, outputs=[chat_input_box], api_name=False
            )
        )
    # -----------------------------------------------

# --- Launch the App ---
if __name__ == "__main__":
    server_name = getattr(config, 'GRADIO_SERVER_NAME', '0.0.0.0')
    server_port = getattr(config, 'GRADIO_SERVER_PORT', 7860)
    log.info(f"Launching Naini Gradio App on http://{server_name}:{server_port} (V5.9 - User Uploads Enabled)")
    # Use queue() for handling multiple requests / long processes like uploads
    demo.queue().launch(server_name=server_name, server_port=server_port, share=False)
    log.info("Naini Gradio App shut down.")