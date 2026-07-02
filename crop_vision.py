"""
ZaminAI Crop Vision — YOLO plant disease detector
==================================================
Supports YOLOv8 and YOLO11 (ultralytics >= 8.3.0).

Model selection (environment variables):
  YOLO_MODEL_PATH  — absolute path to a local .pt file (highest priority)
  YOLO_MODEL_REPO  — HuggingFace repo ID  (default: keremberke/yolov8m-plant-disease)
  YOLO_MODEL_FILE  — filename inside that repo (default: best.pt)
  YOLO_CONF        — confidence threshold (default: 0.25)
  YOLO_IOU         — NMS IoU threshold    (default: 0.45)
  DISABLE_YOLO     — set to "1" to skip YOLO entirely

Default model: keremberke/yolov8m-plant-disease (YOLOv8m, 38 classes, ~25 MB)
Swap to YOLO11:  YOLO_MODEL_REPO=Ultralytics/assets (general) or any fine-tuned repo
"""
import io, logging, os
from pathlib import Path

log = logging.getLogger(__name__)

# ── Config from env ──────────────────────────────────────────────────────────
_CUSTOM_PATH  = os.environ.get("YOLO_MODEL_PATH", "").strip()
_MODEL_REPO   = os.environ.get("YOLO_MODEL_REPO", "keremberke/yolov8m-plant-disease")
_MODEL_FILE   = os.environ.get("YOLO_MODEL_FILE", "best.pt")
_CONF_THR     = float(os.environ.get("YOLO_CONF", "0.25"))
_IOU_THR      = float(os.environ.get("YOLO_IOU",  "0.45"))

_MODEL_DIR    = Path(__file__).parent / "models"
_MODEL_PATH   = Path(_CUSTOM_PATH) if _CUSTOM_PATH else (_MODEL_DIR / "plant_disease.pt")

_yolo_model   = None
_yolo_ok      = False
_yolo_tried   = False

# ── Disease / crop class labels ───────────────────────────────────────────────
# PlantVillage 38 classes (base model) + extended global crop diseases
CLASS_NAMES = {
    # Apple
    "Apple___Apple_scab":                                  "Apple Scab",
    "Apple___Black_rot":                                   "Apple Black Rot",
    "Apple___Cedar_apple_rust":                            "Cedar Apple Rust",
    "Apple___healthy":                                     "Healthy Apple",
    # Berry
    "Blueberry___healthy":                                 "Healthy Blueberry",
    "Raspberry___healthy":                                 "Healthy Raspberry",
    "Strawberry___Leaf_scorch":                            "Strawberry Leaf Scorch",
    "Strawberry___healthy":                                "Healthy Strawberry",
    # Cherry
    "Cherry_(including_sour)___Powdery_mildew":            "Cherry Powdery Mildew",
    "Cherry_(including_sour)___healthy":                   "Healthy Cherry",
    # Corn / Maize
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot":  "Corn Gray Leaf Spot",
    "Corn_(maize)___Common_rust_":                         "Corn Common Rust",
    "Corn_(maize)___Northern_Leaf_Blight":                 "Corn Northern Leaf Blight",
    "Corn_(maize)___healthy":                              "Healthy Corn",
    # Grape
    "Grape___Black_rot":                                   "Grape Black Rot",
    "Grape___Esca_(Black_Measles)":                        "Grape Black Measles",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)":          "Grape Leaf Blight",
    "Grape___healthy":                                     "Healthy Grape",
    # Citrus
    "Orange___Haunglongbing_(Citrus_greening)":            "Citrus Greening Disease",
    # Peach
    "Peach___Bacterial_spot":                              "Peach Bacterial Spot",
    "Peach___healthy":                                     "Healthy Peach",
    # Pepper
    "Pepper,_bell___Bacterial_spot":                       "Pepper Bacterial Spot",
    "Pepper,_bell___healthy":                              "Healthy Pepper",
    # Potato
    "Potato___Early_blight":                               "Potato Early Blight",
    "Potato___Late_blight":                                "Potato Late Blight (Phytophthora)",
    "Potato___healthy":                                    "Healthy Potato",
    # Soybean
    "Soybean___healthy":                                   "Healthy Soybean",
    # Squash
    "Squash___Powdery_mildew":                             "Squash Powdery Mildew",
    # Tomato
    "Tomato___Bacterial_spot":                             "Tomato Bacterial Spot",
    "Tomato___Early_blight":                               "Tomato Early Blight (Alternaria)",
    "Tomato___Late_blight":                                "Tomato Late Blight (Phytophthora)",
    "Tomato___Leaf_Mold":                                  "Tomato Leaf Mold",
    "Tomato___Septoria_leaf_spot":                         "Tomato Septoria Leaf Spot",
    "Tomato___Spider_mites Two-spotted_spider_mite":       "Tomato Spider Mites",
    "Tomato___Target_Spot":                                "Tomato Target Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus":              "Tomato Yellow Leaf Curl Virus",
    "Tomato___Tomato_mosaic_virus":                        "Tomato Mosaic Virus",
    "Tomato___healthy":                                    "Healthy Tomato",
    # ── Extended global crop classes (loaded by alternative models) ──────────
    # Wheat
    "Wheat___Yellow_rust":                                 "Wheat Yellow Rust (Stripe Rust)",
    "Wheat___Brown_rust":                                  "Wheat Brown Rust (Leaf Rust)",
    "Wheat___Stem_rust":                                   "Wheat Stem Rust",
    "Wheat___Powdery_mildew":                              "Wheat Powdery Mildew",
    "Wheat___Septoria":                                    "Wheat Septoria Leaf Blotch",
    "Wheat___Fusarium_head_blight":                        "Wheat Fusarium Head Blight",
    "Wheat___healthy":                                     "Healthy Wheat",
    # Rice
    "Rice___Blast":                                        "Rice Blast (Magnaporthe)",
    "Rice___Brown_spot":                                   "Rice Brown Spot",
    "Rice___Bacterial_blight":                             "Rice Bacterial Leaf Blight",
    "Rice___Sheath_blight":                                "Rice Sheath Blight",
    "Rice___healthy":                                      "Healthy Rice",
    # Cassava (Africa)
    "Cassava___Brown_streak_disease":                      "Cassava Brown Streak Disease",
    "Cassava___Mosaic_disease":                            "Cassava Mosaic Disease",
    "Cassava___Green_mite":                                "Cassava Green Mite",
    "Cassava___Bacterial_blight":                          "Cassava Bacterial Blight",
    "Cassava___healthy":                                   "Healthy Cassava",
    # Coffee
    "Coffee___Leaf_rust":                                  "Coffee Leaf Rust (La Roya)",
    "Coffee___healthy":                                    "Healthy Coffee",
    # Sorghum
    "Sorghum___Anthracnose":                               "Sorghum Anthracnose",
    "Sorghum___healthy":                                   "Healthy Sorghum",
}


def load_model():
    """Lazy-load YOLO model. Downloads from HuggingFace on first use if needed."""
    global _yolo_model, _yolo_ok, _yolo_tried
    if _yolo_tried:
        return _yolo_model
    _yolo_tried = True
    try:
        from ultralytics import YOLO
        import ultralytics
        uv = getattr(ultralytics, "__version__", "unknown")
        log.info(f"ultralytics {uv} loaded")

        _MODEL_DIR.mkdir(exist_ok=True)

        if not _MODEL_PATH.exists():
            log.info(f"Downloading {_MODEL_REPO}/{_MODEL_FILE} from HuggingFace (~25 MB)...")
            try:
                from huggingface_hub import hf_hub_download
                src = hf_hub_download(
                    repo_id=_MODEL_REPO,
                    filename=_MODEL_FILE,
                    local_dir=str(_MODEL_DIR),
                    local_dir_use_symlinks=False,
                )
                import shutil
                shutil.copy(src, str(_MODEL_PATH))
                log.info(f"Model saved to {_MODEL_PATH}")
            except Exception as dl_err:
                log.error(f"Model download failed: {dl_err}")
                return None

        _yolo_model = YOLO(str(_MODEL_PATH))
        _yolo_ok    = True

        # Detect architecture from model metadata
        arch = "YOLO"
        try:
            arch = _yolo_model.info(verbose=False)[0] if hasattr(_yolo_model, "info") else "YOLO"
        except Exception:
            pass
        nc = len(_yolo_model.names) if hasattr(_yolo_model, "names") else "?"
        log.info(f"✓ YOLO model loaded — {nc} classes — repo: {_MODEL_REPO}")

    except ImportError:
        log.warning("ultralytics not installed — YOLO disabled, Claude Vision handles diagnosis")
    except MemoryError:
        log.error("YOLO load: out of memory — set DISABLE_YOLO=1 on this host")
    except Exception as e:
        log.error(f"YOLO model load failed: {e}")
    return _yolo_model


def run_inference(image_bytes: bytes) -> dict:
    """
    Run YOLO inference on raw image bytes.
    Returns {ok, yolo_available, model_repo, detections: [{label, label_en, confidence, bbox, is_healthy}]}
    """
    model = load_model()
    if model is None:
        return {"ok": False, "yolo_available": False, "detections": [], "model_repo": _MODEL_REPO}
    try:
        from PIL import Image
        img     = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        results = model.predict(img, conf=_CONF_THR, iou=_IOU_THR, verbose=False)

        detections = []
        for r in results:
            boxes = r.boxes if hasattr(r, "boxes") else []
            for box in boxes:
                raw_label = model.names[int(box.cls[0])]
                conf      = float(box.conf[0])
                en_label  = CLASS_NAMES.get(raw_label, raw_label.replace("_", " ").replace("___", " — "))
                detections.append({
                    "label":      raw_label,
                    "label_en":   en_label,
                    "confidence": round(conf, 3),
                    "bbox":       [round(v, 1) for v in box.xyxy[0].tolist()],
                    "is_healthy": "healthy" in raw_label.lower(),
                })

            # YOLO11 classification task returns probs instead of boxes
            if hasattr(r, "probs") and r.probs is not None:
                top5_i   = r.probs.top5
                top5_c   = r.probs.top5conf.tolist()
                for idx, conf in zip(top5_i, top5_c):
                    if conf < _CONF_THR:
                        break
                    raw_label = model.names[idx]
                    en_label  = CLASS_NAMES.get(raw_label, raw_label.replace("_", " "))
                    detections.append({
                        "label":      raw_label,
                        "label_en":   en_label,
                        "confidence": round(float(conf), 3),
                        "bbox":       [],
                        "is_healthy": "healthy" in raw_label.lower(),
                    })

        detections.sort(key=lambda x: x["confidence"], reverse=True)
        return {
            "ok":           True,
            "yolo_available": True,
            "model_repo":   _MODEL_REPO,
            "detections":   detections,
        }
    except MemoryError:
        log.error("YOLO inference OOM")
        return {"ok": False, "yolo_available": True, "error": "OOM", "detections": [], "model_repo": _MODEL_REPO}
    except Exception as e:
        log.error(f"YOLO inference error: {e}")
        return {"ok": False, "yolo_available": True, "error": str(e), "detections": [], "model_repo": _MODEL_REPO}
