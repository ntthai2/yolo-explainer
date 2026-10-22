"""
FastAPI Server for YOLO Explainer.
Exposes REST endpoints for model exploration, layer activation inspection,
interactive feature map generation, and static UI delivery.
"""

import os
import io
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from PIL import Image

from backend.app.core.model_registry import registry, MODEL_CONFIGS
from backend.app.core.extractor import extract_features_and_detect, preprocess_image
from backend.app.core.visualizer import (
    process_layer_activation,
    render_detections_image,
    render_candidates_image,
    array_to_base64_jpeg
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
STATIC_DIR = os.path.join(BASE_DIR, "backend", "app", "static")
SAMPLES_DIR = os.path.join(STATIC_DIR, "samples")

app = FastAPI(
    title="YOLO Explainer API",
    description="Interactive Web Application to explore YOLO architecture evolution (v8n, 11n, 26n)",
    version="1.0.0"
)

# Enable GZip compression for fast JSON transfers
app.add_middleware(
    GZipMiddleware,
    minimum_size=1000
)

# Enable CORS for local/cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory cache for precomputed samples: key = f"{model_name}:{sample_id}"
CACHE: Dict[str, Dict[str, Any]] = {}


def compute_inspect_data(model_name: str, pil_image: Image.Image) -> Dict[str, Any]:
    """Chạy trích xuất đặc trưng và trực quan hóa các tầng."""
    res = extract_features_and_detect(model_name, pil_image)
    activations = res["activations"]
    layer_info_map = res["layer_info_map"]
    preprocessed_rgb = res["preprocessed_rgb"]
    detections = res["detections"]
    candidates = res.get("candidates", [])
    nms_stats = res.get("nms_stats", {})
    inference_time = res["inference_time_ms"]

    # Preprocessed original image
    original_b64 = array_to_base64_jpeg(preprocessed_rgb, quality=80)
    
    # Annotated detection image (Post-NMS clean)
    detection_img_b64 = render_detections_image(preprocessed_rgb, detections)
    # Annotated candidate image (Pre-NMS glowing clusters)
    candidates_img_b64 = render_candidates_image(preprocessed_rgb, candidates, detections)

    # Process each layer activations
    processed_layers: Dict[str, Any] = {}
    for layer_id, tensor in activations.items():
        if layer_id in layer_info_map:
            layer_meta = layer_info_map[layer_id]
            viz_data = process_layer_activation(
                activation=tensor,
                layer_meta=layer_meta,
                original_rgb=preprocessed_rgb,
                top_k_channels=12
            )
            processed_layers[layer_id] = viz_data

    return {
        "model_name": model_name,
        "inference_time_ms": inference_time,
        "image_info": {
            "width": preprocessed_rgb.shape[1],
            "height": preprocessed_rgb.shape[0]
        },
        "original_image": original_b64,
        "detection_image": detection_img_b64,
        "candidates_image": candidates_img_b64,
        "detections": detections,
        "candidates": candidates,
        "nms_stats": nms_stats,
        "layers": processed_layers,
        "device": registry.device
    }


@app.on_event("startup")
async def startup_event():
    """Khởi động, nạp sẵn các mô hình và làm ấm bộ nhớ đệm cho ảnh mặc định."""
    print("Khởi động và nạp model trên server...")
    bus_path = os.path.join(SAMPLES_DIR, "bus.jpg")
    bus_img = None
    if os.path.exists(bus_path):
        try:
            bus_img = Image.open(bus_path).convert("RGB")
        except Exception:
            pass

    for model_name in MODEL_CONFIGS.keys():
        try:
            print(f"Loading {model_name}...")
            registry.get_model(model_name)
            if bus_img:
                cache_key = f"{model_name}:bus"
                if cache_key not in CACHE:
                    print(f"Làm ấm cache cho {cache_key}...")
                    CACHE[cache_key] = compute_inspect_data(model_name, bus_img)
        except Exception as e:
            print(f"Cảnh báo: không thể nạp sẵn {model_name}: {e}")
    print("Hoàn tất nạp model và làm ấm cache.")


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "device": registry.device,
        "models_available": list(MODEL_CONFIGS.keys())
    }


@app.get("/api/models")
def get_models():
    """Returns list of models and their architectural configuration."""
    return {
        "models": registry.list_all_models(),
        "default_model": "yolo11n"
    }


@app.get("/api/samples")
def get_samples():
    """Lists available pre-loaded sample images."""
    samples = []
    if os.path.exists(SAMPLES_DIR):
        for fname in sorted(os.listdir(SAMPLES_DIR)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                sample_id = os.path.splitext(fname)[0]
                samples.append({
                    "id": sample_id,
                    "filename": fname,
                    "title": sample_id.replace("_", " ").title(),
                    "url": f"/static/samples/{fname}"
                })
    return {"samples": samples}


@app.post("/api/inspect")
async def inspect_model(
    model_name: str = Form("yolo11n"),
    sample_id: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    """
    Main inspection endpoint.
    Accepts model_name and either an uploaded file or sample_id.
    Returns layer activations, heatmaps, top channels, and detections.
    """
    if model_name not in MODEL_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Invalid model_name. Choose from: {list(MODEL_CONFIGS.keys())}")

    # Check cache for sample_id
    cache_key = f"{model_name}:{sample_id}"
    if sample_id and not file and cache_key in CACHE:
        return CACHE[cache_key]

    # Load image
    pil_image = None
    if file:
        try:
            contents = await file.read()
            pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Cannot read image file: {str(e)}")
    elif sample_id:
        sample_path = os.path.join(SAMPLES_DIR, f"{sample_id}.jpg")
        if not os.path.exists(sample_path):
            sample_path = os.path.join(SAMPLES_DIR, f"{sample_id}.png")
        if not os.path.exists(sample_path):
            raise HTTPException(status_code=404, detail=f"Sample '{sample_id}' not found")
        pil_image = Image.open(sample_path).convert("RGB")
    else:
        # Default to first available sample
        sample_path = os.path.join(SAMPLES_DIR, "bus.jpg")
        if os.path.exists(sample_path):
            pil_image = Image.open(sample_path).convert("RGB")
        else:
            raise HTTPException(status_code=400, detail="No image provided and no default sample available")

    # Run extraction & detection
    try:
        response_data = compute_inspect_data(model_name, pil_image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {str(e)}")

    # Store in cache if sample_id
    if sample_id and not file:
        CACHE[cache_key] = response_data

    return response_data


# Mount static samples directory
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Mount frontend directory for web UI
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
