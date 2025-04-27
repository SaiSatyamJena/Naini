# naini_app/layout_detector.py

from transformers import AutoImageProcessor, DetrForSegmentation
import torch
from PIL import Image
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables for processor and model to load them only once
_processor = None
_model = None
_model_name = "cmarkea/detr-layout-detection" # Define model name here

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

def detect_layout_elements(processor, model, image_path, detection_threshold=0.4):
    """
    Detects layout elements in an image using the DETR model.

    Args:
        processor: The loaded AutoImageProcessor object.
        model: The loaded DetrForSegmentation model object.
        image_path: Path to the input image.
        detection_threshold: Confidence threshold for filtering detections.

    Returns:
        A list of dictionaries, where each dictionary represents a detected element
        with keys: 'box' (list [x1, y1, x2, y2]), 'label' (str), 'confidence' (float).
        Returns an empty list if detection fails.
    """
    if not processor or not model:
        logging.error("DETR processor or model not provided for detection.")
        return []
    if not Path(image_path).exists():
        logging.error(f"Image path does not exist for detection: {image_path}")
        return []

    logging.info(f"Running DETR layout detection for: {image_path}")

    try:
        image = Image.open(image_path).convert("RGB")

        # Preprocess image
        inputs = processor(images=image, return_tensors="pt")

        # Placeholder: Move inputs to the same device as the model if using GPU
        # inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Perform inference
        with torch.inference_mode():
            outputs = model(**inputs)

        # Post-process to get bounding boxes
        # target_sizes expects a list of tuples [(height, width)]
        target_sizes = [image.size[::-1]] # PIL size is (width, height), need (height, width)
        results = processor.post_process_object_detection(
            outputs, threshold=detection_threshold, target_sizes=target_sizes
        )[0] # Get results for the first (and only) image

        detections = []
        # Ensure model.config.id2label is available
        if hasattr(model, 'config') and hasattr(model.config, 'id2label'):
            id2label = model.config.id2label
            for score, label_id, box in zip(results["scores"], results["labels"], results["boxes"]):
                label = id2label.get(label_id.item(), "unknown") # Get label name from ID
                detections.append(
                    {
                        "score": score.item(), # DETR uses 'score'
                        "confidence": score.item(), # Add 'confidence' for consistency
                        "label": label,
                        "box": box.tolist(), # Box format [xmin, ymin, xmax, ymax]
                    }
                )
            logging.info(f"Detected {len(detections)} elements in {image_path} with threshold {detection_threshold}")
        else:
             logging.error("Cannot map label IDs to names. model.config.id2label not found.")

        return detections

    except Exception as e:
        logging.error(f"Error during DETR prediction for {image_path}: {e}", exc_info=True)
        return []

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