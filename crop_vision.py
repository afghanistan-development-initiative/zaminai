"""
ZaminAI Crop Vision — YOLO plant disease detector
Model: keremberke/yolov8m-plant-disease (YOLOv8 Medium, 38 disease classes)
Trained on PlantVillage + PlantDoc. Falls back gracefully if ultralytics is absent.
"""
import io, logging
from pathlib import Path

log = logging.getLogger(__name__)

_MODEL_DIR  = Path(__file__).parent / "models"
_MODEL_PATH = _MODEL_DIR / "plant_disease.pt"
_yolo_model = None
_yolo_ok    = False
_yolo_tried = False

# Human-readable labels for all 38 PlantVillage classes
CLASS_NAMES = {
    "Apple___Apple_scab":                                  "Apple Scab",
    "Apple___Black_rot":                                   "Apple Black Rot",
    "Apple___Cedar_apple_rust":                            "Cedar Apple Rust",
    "Apple___healthy":                                     "Healthy Apple",
    "Blueberry___healthy":                                 "Healthy Blueberry",
    "Cherry_(including_sour)___Powdery_mildew":            "Cherry Powdery Mildew",
    "Cherry_(including_sour)___healthy":                   "Healthy Cherry",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot":  "Corn Gray Leaf Spot",
    "Corn_(maize)___Common_rust_":                         "Corn Common Rust",
    "Corn_(maize)___Northern_Leaf_Blight":                 "Corn Northern Leaf Blight",
    "Corn_(maize)___healthy":                              "Healthy Corn",
    "Grape___Black_rot":                                   "Grape Black Rot",
    "Grape___Esca_(Black_Measles)":                        "Grape Black Measles",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)":          "Grape Leaf Blight",
    "Grape___healthy":                                     "Healthy Grape",
    "Orange___Haunglongbing_(Citrus_greening)":            "Citrus Greening",
    "Peach___Bacterial_spot":                              "Peach Bacterial Spot",
    "Peach___healthy":                                     "Healthy Peach",
    "Pepper,_bell___Bacterial_spot":                       "Pepper Bacterial Spot",
    "Pepper,_bell___healthy":                              "Healthy Pepper",
    "Potato___Early_blight":                               "Potato Early Blight",
    "Potato___Late_blight":                                "Potato Late Blight",
    "Potato___healthy":                                    "Healthy Potato",
    "Raspberry___healthy":                                 "Healthy Raspberry",
    "Soybean___healthy":                                   "Healthy Soybean",
    "Squash___Powdery_mildew":                             "Squash Powdery Mildew",
    "Strawberry___Leaf_scorch":                            "Strawberry Leaf Scorch",
    "Strawberry___healthy":                                "Healthy Strawberry",
    "Tomato___Bacterial_spot":                             "Tomato Bacterial Spot",
    "Tomato___Early_blight":                               "Tomato Early Blight",
    "Tomato___Late_blight":                                "Tomato Late Blight",
    "Tomato___Leaf_Mold":                                  "Tomato Leaf Mold",
    "Tomato___Septoria_leaf_spot":                         "Tomato Septoria Leaf Spot",
    "Tomato___Spider_mites Two-spotted_spider_mite":       "Tomato Spider Mites",
    "Tomato___Target_Spot":                                "Tomato Target Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":              "Tomato Yellow Leaf Curl Virus",
    "Tomato___Tomato_mosaic_virus":                        "Tomato Mosaic Virus",
    "Tomato___healthy":                                    "Healthy Tomato",
}


def load_model():
    """Lazy-load the YOLO model. Download from HuggingFace on first use."""
    global _yolo_model, _yolo_ok, _yolo_tried
    if _yolo_tried:
        return _yolo_model
    _yolo_tried = True
    try:
        from ultralytics import YOLO
        _MODEL_DIR.mkdir(exist_ok=True)
        if not _MODEL_PATH.exists():
            log.info("Downloading plant disease model from HuggingFace Hub (~25 MB)...")
            from huggingface_hub import hf_hub_download
            src = hf_hub_download(
                repo_id="keremberke/yolov8m-plant-disease",
                filename="best.pt",
                local_dir=str(_MODEL_DIR),
                local_dir_use_symlinks=False,
            )
            import shutil
            shutil.copy(src, str(_MODEL_PATH))
        _yolo_model = YOLO(str(_MODEL_PATH))
        _yolo_ok = True
        log.info("✓ YOLO plant disease model loaded (YOLOv8m, 38 classes)")
    except ImportError:
        log.warning("ultralytics not installed — YOLO disabled, Claude Vision will handle diagnosis")
    except Exception as e:
        log.error(f"YOLO model load failed: {e}")
    return _yolo_model


def run_inference(image_bytes: bytes) -> dict:
    """
    Run YOLO on raw image bytes.
    Returns {"ok", "yolo_available", "detections": [{label, label_en, confidence, bbox, is_healthy}]}
    """
    model = load_model()
    if model is None:
        return {"ok": False, "yolo_available": False, "detections": []}
    try:
        from PIL import Image
        img     = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        results = model.predict(img, conf=0.25, iou=0.45, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                label = model.names[int(box.cls[0])]
                conf  = float(box.conf[0])
                detections.append({
                    "label":      label,
                    "label_en":   CLASS_NAMES.get(label, label.replace("_", " ")),
                    "confidence": round(conf, 3),
                    "bbox":       [round(v, 1) for v in box.xyxy[0].tolist()],
                    "is_healthy": "healthy" in label.lower(),
                })
        detections.sort(key=lambda x: x["confidence"], reverse=True)
        return {"ok": True, "yolo_available": True, "detections": detections}
    except Exception as e:
        log.error(f"YOLO inference error: {e}")
        return {"ok": False, "yolo_available": True, "error": str(e), "detections": []}
