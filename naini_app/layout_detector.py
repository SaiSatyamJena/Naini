# naini_app/layout_detector.py

from transformers import AutoImageProcessor, DetrForSegmentation
import torch
from PIL import Image
import logging
from pathlib import Path
from transformers import pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables for processor and model to load them only once
_processor = None
_model = None
_model_name = "microsoft/table-transformer-detection" # Define model name here

def load_layout_model_and_processor(model_name=_model_name):
    """Loads the DETR model and its image processor."""
    global _processor, _model
    if _processor and _model:
        logging.info(f"Model '{model_name}' and processor already loaded.")
        return _processor, _model

    logging.info(f"Loading DETR model and processor: {model_name}")
    try:
        _processor = AutoImageProcessor.from_pretrained(model_name)
        _model = DetrForSegmentation.from_pretrained(model_name)

        # --- Log class names (stored differently in transformers models) ---
        if hasattr(_model, 'config') and hasattr(_model.config, 'id2label'):
             logging.info(f"--- ACTUAL MODEL CLASS NAMES (id2label): {_model.config.id2label} ---")
        else:
             logging.warning("Could not automatically determine model class names (id2label).")
        # --- End class name logging ---

        # Placeholder: Move model to GPU if available and configured
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # _model.to(device)
        # logging.info(f"Model moved to device: {device}")

        logging.info(f"DETR Model '{model_name}' and processor loaded successfully.")
        return _processor, _model

    except Exception as e:
        logging.error(f"Error loading DETR model/processor {model_name}: {e}", exc_info=True)
        _processor, _model = None, None # Reset on failure
        return None, None

def detect_layout_elements(processor, model, image_path, detection_threshold=0.8): # << INCREASED Threshold slightly
    """
    Detects layout elements including tables in an image using a pipeline.
    Now adapted for models like microsoft/table-transformer-detection.

    Args:
        processor: The loaded processor (may not be strictly needed by pipeline, but passed for consistency).
        model: The loaded model (may not be strictly needed by pipeline, but passed for consistency).
        image_path: Path to the input image.
        detection_threshold: Confidence threshold for filtering detections (0.0 to 1.0).
                            Table Transformers often require higher thresholds (e.g., 0.7-0.9).

    Returns:
        A list of dictionaries, where each dictionary represents a detected element
        with keys: 'box' (list [x1, y1, x2, y2]), 'label' (str), 'score' (float).
        Returns an empty list if detection fails. Returns original labels from the model.
    """
    # Note: This pipeline approach simplifies handling different detection models.
    # It might load the model again internally if not perfectly cached, but ensures compatibility.
    # We pass processor/model mainly to check they were loaded initially.
    if not processor or not model:
         log.error("DETR processor or model not provided/loaded for detection.")
         return []
    if not Path(image_path).exists():
         log.error(f"Image path does not exist for detection: {image_path}")
         return []

    # <<< USING PIPELINE INSTEAD OF MANUAL POST-PROCESSING >>>
    try:
         log.info(f"Running Object Detection pipeline for: {image_path} using model {_model_name}")
         # Use the object-detection pipeline directly with the specified model
         object_detector = pipeline("object-detection", model=_model_name)

         # Perform inference
         results = object_detector(image_path) # Pipeline handles image loading/preprocessing

         # Filter results based on the threshold and format output
         detections = []
         log.info(f"Raw pipeline detections count: {len(results)}") # Log how many raw detections
         for detection in results:
              # Example detection: {'score': 0.999, 'label': 'table', 'box': {'xmin': 10, 'ymin': 20, 'xmax': 100, 'ymax': 200}}
              score = detection.get('score', 0.0)
              label = detection.get('label', 'unknown').lower() # Use lower case label
              box_dict = detection.get('box', {})

              # <<< CRITICAL LABEL MAPPING (Example) >>>
              # Map Table Transformer specific labels if needed
              if label == 'table':
                   pass # Keep 'table' label
              elif label == 'table column': # Often detected, maybe useful later but skip for now
                  log.debug(f"Skipping 'table column' detection: {detection}")
                  continue
              elif label == 'table row': # Often detected, maybe useful later but skip for now
                   log.debug(f"Skipping 'table row' detection: {detection}")
                   continue
              # Add other label mappings or filtering if the new model has different output classes
              # e.g., map 'text' to 'Text', 'figure' to 'Picture', etc.

              # Apply threshold filter
              if score >= detection_threshold and box_dict:
                    formatted_box = [
                        box_dict.get('xmin', 0),
                        box_dict.get('ymin', 0),
                        box_dict.get('xmax', 0),
                        box_dict.get('ymax', 0)
                    ]
                    detections.append(
                        {
                            "score": score, # Keep 'score' from pipeline
                            "label": label, # Use the (potentially mapped) label
                            "box": formatted_box, # Standardize box format
                        }
                    )
                    log.debug(f"  KEEPING detection: Score={score:.3f}, Label='{label}', Box={formatted_box}")
              else:
                    log.debug(f"  FILTERING detection: Score={score:.3f} (<{detection_threshold}), Label='{label}', Box={box_dict}")


         log.info(f"Detected and kept {len(detections)} elements in {image_path} with threshold {detection_threshold}")
         return detections

    except Exception as e:
         log.error(f"Error during object detection pipeline for {image_path}: {e}", exc_info=True)
         return []
# --- END REPLACEMENT ---

if __name__ == '__main__':
    print("Testing layout_detector.py with DETR model...")
    processor, model = load_layout_model_and_processor() # Load the default DETR model

    if processor and model:
        # Use a real image generated previously
        test_image = "data/output_markdown/run_1/page_images/page_0001-107.png" # Annexure 2 table page
        if Path(test_image).exists():
             print(f"Running detection on test image: {test_image}")
             detections = detect_layout_elements(processor, model, test_image)
             print(f"Detected {len(detections)} elements (showing all):")
             for det in detections:
                 conf = round(det['confidence'], 2)
                 box = [round(c) for c in det['box']] # Round coordinates
                 print(f"  Label: {det['label']}, Confidence: {conf}, Box: {box}")
             if not detections:
                  print("  No elements detected on this test page.")
        else:
             print(f"Need a test image at {test_image} to run detection test. Generate images first.")
    else:
        print("DETR model or processor loading failed.")