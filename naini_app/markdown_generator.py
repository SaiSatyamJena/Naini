# naini_app/markdown_generator.py
import logging
import re
import difflib # Import difflib for fuzzy duplicate matching

log = logging.getLogger(__name__)

# --- Constants ---
TABLE_FAILURE_PLACEHOLDER = "[TABLE EXTRACTION FAILED OR EMPTY]"
VERTICAL_PROXIMITY_THRESHOLD_LIST = 20 # Max vertical pixels between list items (Value from Run 20)
DUPLICATE_SIMILARITY_THRESHOLD = 0.95 # Similarity ratio for fuzzy duplicate removal

# --- Regex Patterns ---
ALLOWED_CHAR_PATTERN = re.compile(
    r"[^\u0020-\u007E\u00A0-\u00FF\u2000-\u206F\u2190-\u21FF\u2200-\u22FF\u00B0\u00B1\u00BC-\u00BE\u2150-\u215E]+"
)
GENERAL_CLEANUP_PATTERN = re.compile(
    r"([¦¬])"
    r"|((?<!\S)[.,!?;:'](?!\S))"
)

# --- Helper Function: Filter Non-English/Undesired Characters ---
# Defined HERE in this version
def _filter_non_english(text: str) -> str:
    """Removes characters outside defined allowed ranges (e.g., Hindi script)."""
    if not text: return ""
    filtered_text = ALLOWED_CHAR_PATTERN.sub('', text)
    return re.sub(r'\s+', ' ', filtered_text).strip()

# --- Helper Function: General OCR Cleanup ---
# Defined HERE in this version
def _general_cleanup(text: str) -> str:
    """Removes common OCR noise characters after script filtering."""
    if not text: return ""
    cleaned_text = GENERAL_CLEANUP_PATTERN.sub('', text)
    return re.sub(r'\s+', ' ', cleaned_text).strip()

# --- Helper Function: Post-Process Stream Tables ---
def _post_process_stream_table(md_table_string: str) -> str:
    # (Unchanged - Same as previous versions)
    if not md_table_string or md_table_string == TABLE_FAILURE_PLACEHOLDER: return md_table_string
    log.debug("Post-processing Camelot Stream table...")
    lines = md_table_string.strip().split('\n')
    if len(lines) < 2: log.warning("Table too short for post-processing."); return md_table_string
    header_line, separator_line, data_lines = lines[0], lines[1], lines[2:]
    parse_line = lambda line: [cell.strip() for cell in line.strip().strip('|').split('|')]
    try:
        header_cells = parse_line(header_line)
        separator_cells = parse_line(separator_line)
        num_cols = len(header_cells)
        if num_cols == 0 or num_cols != len(separator_cells): log.warning(f"Inconsistent cols: H({len(header_cells)}) S({len(separator_cells)}). Skip."); return md_table_string
        data_rows = [parse_line(line) for line in data_lines]
        for i, row in enumerate(data_rows):
            if len(row) != num_cols: log.warning(f"Row {i+3} bad cols: {len(row)}/{num_cols}. Skip."); return md_table_string
        empty_col_indices = set()
        for col_idx in range(num_cols):
            is_empty = all(not row[col_idx].strip() for row in data_rows if col_idx < len(row))
            if is_empty: log.debug(f"Empty col index {col_idx}"); empty_col_indices.add(col_idx)
        if not empty_col_indices: log.debug("No empty cols."); return md_table_string
        new_lines = []
        new_header_cells = [c for i, c in enumerate(header_cells) if i not in empty_col_indices]
        new_lines.append("| " + " | ".join(new_header_cells) + " |")
        new_separator_cells = [c for i, c in enumerate(separator_cells) if i not in empty_col_indices]
        new_lines.append("|" + "|".join(new_separator_cells) + "|")
        for row in data_rows:
            new_row_cells = [c for i, c in enumerate(row) if i not in empty_col_indices]
            new_lines.append("| " + " | ".join(new_row_cells) + " |")
        cleaned_md = "\n".join(new_lines)
        log.info(f"Removed {len(empty_col_indices)} empty columns.")
        return cleaned_md
    except Exception as e: log.error(f"Table post-proc error: {e}", exc_info=True); return md_table_string

# --- Main Markdown Generation Function (No Paragraph Reconstruction) ---
def generate_markdown_page(detections, page_number):
    log.info(f"Generating Markdown for page {page_number} with {len(detections)} detections.")

    # --- 1. Initial Sort ---
    try:
        valid_detections = [d for d in detections if isinstance(d.get('box'), (list, tuple)) and len(d['box']) == 4]
        if len(valid_detections) < len(detections): log.warning(f"Removed {len(detections)-len(valid_detections)} detections without valid 'box'.")
        valid_detections.sort(key=lambda d: (d['box'][1], d['box'][0]))
        sorted_detections = valid_detections
    except Exception as e:
        log.error(f"Sorting error page {page_number}: {e}. Using original order.", exc_info=True)
        sorted_detections = [{'box': d.get('box', [0,0,0,0]), **d} for d in detections]

    # --- 2. Process and Format Each Detection ---
    formatted_blocks = []
    for i, det in enumerate(sorted_detections):
        label = det.get('label', 'unknown')
        box = det.get('box')
        y1 = box[1] if box else float('inf')
        y2 = box[3] if box else float('inf')
        formatted_text = ""

        if label in ['Page-header', 'Page-footer']: log.debug(f"Ignoring '{label}' pg {page_number}."); continue

        elif label == 'Table':
            table_info = det.get('table_data', (TABLE_FAILURE_PLACEHOLDER, 'failure'))
            if isinstance(table_info, str): table_md, method_used = table_info, 'unknown'
            elif isinstance(table_info, tuple) and len(table_info) == 2: table_md, method_used = table_info
            else: table_md, method_used = TABLE_FAILURE_PLACEHOLDER, 'failure'

            content_to_append = table_md if table_md else TABLE_FAILURE_PLACEHOLDER
            if method_used == 'camelot-stream' and content_to_append != TABLE_FAILURE_PLACEHOLDER:
                content_to_append = _post_process_stream_table(content_to_append)
            formatted_text = f"{content_to_append}\n\n"

        elif label == 'Picture':
            formatted_text = "[FIGURE DETECTED: Picture]\n\n"

        else: # Text-based labels
            raw_text = det.get('text', '')
            # Apply filters HERE
            filtered_script_text = _filter_non_english(raw_text)
            cleaned_text = _general_cleanup(filtered_script_text)
            if not cleaned_text: log.debug(f"Skipping empty block '{label}' pg {page_number}."); continue

            # Apply semantic formatting
            if label == 'Title': formatted_text = f"# {cleaned_text}\n\n";
            elif label == 'Section-header': formatted_text = f"## {cleaned_text}\n\n";
            elif label == 'List-item': formatted_text = f"- {cleaned_text}\n"; # Single newline for grouping
            elif label == 'Caption': formatted_text = f"*Caption: {cleaned_text}*\n\n";
            elif label == 'Formula': formatted_text = f"```\n{cleaned_text}\n```\n\n";
            elif label == 'Footnote': formatted_text = f"[^footnote]: {cleaned_text}\n\n";
            elif label == 'Text': formatted_text = f"{cleaned_text}\n\n";
            else: log.warning(f"Unmapped label '{label}' pg {page_number}. Defaulting to text."); formatted_text = f"{cleaned_text}\n\n";

        if formatted_text:
             formatted_blocks.append({'y1': y1, 'y2': y2, 'label': label, 'md': formatted_text})

    # --- 3. Post-Processing: Grouping List Items ---
    if not formatted_blocks: return f"## Page {page_number}\n\n[NO CONTENT DETECTED OR ALL FILTERED]\n\n"

    merged_blocks = []
    i = 0
    while i < len(formatted_blocks):
        current_block = formatted_blocks[i]
        if current_block['label'] == 'List-item':
            current_list_items_md = [current_block['md']]
            max_y2 = current_block['y2']
            j = i + 1
            while j < len(formatted_blocks) and \
                  formatted_blocks[j]['label'] == 'List-item' and \
                  (formatted_blocks[j]['y1'] - max_y2) <= VERTICAL_PROXIMITY_THRESHOLD_LIST:
                current_list_items_md.append(formatted_blocks[j]['md'])
                max_y2 = formatted_blocks[j]['y2']
                j += 1
            merged_list_md = "".join(current_list_items_md) + "\n"
            merged_blocks.append({'label': 'List-group', 'md': merged_list_md})
            log.debug(f"Grouped {len(current_list_items_md)} list items on page {page_number}.")
            i = j
        else:
            merged_blocks.append({'label': current_block['label'], 'md': current_block['md']})
            i += 1

    # --- 4. Post-Processing: Remove Similar Duplicate Consecutive Blocks ---
    if not merged_blocks: return f"## Page {page_number}\n\n[NO CONTENT DETECTED OR ALL FILTERED]\n\n"

    final_blocks = []
    duplicate_count = 0
    for i, current_block in enumerate(merged_blocks):
        if not final_blocks: final_blocks.append(current_block); continue
        previous_block = final_blocks[-1]
        current_md_stripped = current_block['md'].strip()
        previous_md_stripped = previous_block['md'].strip()

        # Check content non-empty and minimum length before fuzzy matching
        if not current_md_stripped or not previous_md_stripped or len(current_md_stripped)<15 or len(previous_md_stripped)<15:
             final_blocks.append(current_block)
             continue

        similarity_ratio = difflib.SequenceMatcher(None, previous_md_stripped, current_md_stripped).ratio()
        # Exclude exact TABLE duplicates based on placeholder content
        is_table_placeholder_dup = (current_block['label'] == 'Table' and previous_block['label'] == 'Table' and current_md_stripped == TABLE_FAILURE_PLACEHOLDER and previous_md_stripped == TABLE_FAILURE_PLACEHOLDER)

        if not is_table_placeholder_dup and \
           current_block['label'] == previous_block['label'] and \
           similarity_ratio > DUPLICATE_SIMILARITY_THRESHOLD:
            duplicate_count += 1
            log.debug(f"Near-duplicate detected (Sim: {similarity_ratio:.2f}), Label: {current_block['label']}, Content: '{current_md_stripped[:50]}...' - REMOVING")
        else:
            final_blocks.append(current_block)
    if duplicate_count > 0: log.info(f"Removed {duplicate_count} near-duplicate blocks on page {page_number}.")

    # --- 5. Assemble Final Page Markdown ---
    final_markdown_content = f"## Page {page_number}\n\n"
    if not final_blocks: return f"## Page {page_number}\n\n[NO CONTENT DETECTED OR ALL FILTERED]\n\n"
    for block in final_blocks:
        final_markdown_content += block['md']

    return final_markdown_content.strip() + "\n\n"

# --- Main Block for Standalone Testing ---
if __name__ == '__main__':
     logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s [%(module)s] - %(message)s')
     print("Testing markdown_generator.py (Restored State - Run 20 equivalent)...")
     # Add relevant test cases here if needed, remembering filters are applied internally now.