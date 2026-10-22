"""
PyTorch Hook Feature Extractor for YOLO Explainer.
Safely attaches forward hooks to capture intermediate feature maps across
Backbone, Neck, and Head, and collects final object detections.
"""

import time
from typing import Dict, Any, Tuple, List
import numpy as np
from PIL import Image
import torch
import cv2
from ultralytics import YOLO
from backend.app.core.model_registry import registry, MODEL_CONFIGS


def preprocess_image(image: Image.Image, target_size: int = 640) -> Tuple[torch.Tensor, np.ndarray, float, Tuple[int, int]]:
    """
    Resizes image with aspect ratio letterboxing to target_size x target_size.
    Returns:
        tensor: (1, 3, target_size, target_size) normalized [0, 1] on registry.device
        img_rgb: (target_size, target_size, 3) uint8 numpy array for visualizer blending
        ratio: scale ratio
        pad: (dw, dh) padding
    """
    img_cv = np.array(image.convert("RGB"))
    h, w = img_cv.shape[:2]
    
    # Calculate scale and padding
    r = min(target_size / h, target_size / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw, dh = target_size - new_unpad[0], target_size - new_unpad[1]
    dw /= 2
    dh /= 2

    if (w, h) != new_unpad:
        img_resized = cv2.resize(img_cv, new_unpad, interpolation=cv2.INTER_LINEAR)
    else:
        img_resized = img_cv

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    # Convert to Tensor (1, 3, H, W)
    img_tensor = img_padded.transpose((2, 0, 1)) # HWC to CHW
    img_tensor = np.ascontiguousarray(img_tensor, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(img_tensor).unsqueeze(0).to(registry.device)

    return tensor, img_padded, r, (dw, dh)


from ultralytics.utils import nms


def extract_features_and_detect(
    model_name: str,
    image: Image.Image,
    conf_threshold: float = 0.25
) -> Dict[str, Any]:
    """
    Performs inference with forward hooks on monitored layers.
    Returns extracted activations, pre-NMS candidates, post-NMS detections, and scale attribution.
    """
    yolo_model = registry.get_model(model_name)
    model = yolo_model.model
    model.eval()

    tensor, img_rgb, ratio, (dw, dh) = preprocess_image(image, target_size=640)
    meta = registry.get_metadata(model_name)
    monitored = meta["monitored_layers"]

    activations: Dict[str, torch.Tensor] = {}
    layer_info_map: Dict[str, Dict[str, Any]] = {}
    hooks = []

    def make_hook(layer_id: str):
        def hook_fn(module, input_t, output_t):
            t = output_t
            if isinstance(t, (list, tuple)):
                # If tuple/list, take first tensor with 4 dimensions (B, C, H, W)
                found = False
                for item in t:
                    if isinstance(item, torch.Tensor) and item.dim() == 4:
                        t = item
                        found = True
                        break
                if not found and len(t) > 0 and isinstance(t[0], torch.Tensor):
                    t = t[0]
            if isinstance(t, torch.Tensor):
                # Detach and keep on CPU
                activations[layer_id] = t.detach().cpu()
        return hook_fn

    # Register hooks
    for layer_cfg in monitored:
        idx = layer_cfg["layer_idx"]
        layer_id = layer_cfg["id"]
        if idx < len(model.model):
            submodule = model.model[idx]
            h = submodule.register_forward_hook(make_hook(layer_id))
            hooks.append(h)
            layer_info_map[layer_id] = layer_cfg

    start_time = time.perf_counter()
    try:
        with torch.no_grad():
            preds = model(tensor)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    finally:
        # Guarantee hook cleanup
        for h in hooks:
            h.remove()

    infer_duration_ms = (time.perf_counter() - start_time) * 1000

    out = preds[0] if isinstance(preds, (list, tuple)) else preds
    names = getattr(yolo_model, "names", getattr(model, "names", {}))

    detections: List[Dict[str, Any]] = []
    candidates: List[Dict[str, Any]] = []
    is_nms_free = False

    # Check output format
    if isinstance(out, torch.Tensor) and out.shape[-1] == 8400: # YOLOv8 & YOLO11
        # 1. Raw candidates before NMS
        t_out = out[0].transpose(0, 1) # [8400, 84]
        boxes_cxcywh = t_out[:, :4]
        scores, clss = t_out[:, 4:].max(dim=-1)
        # 1. Raw candidates before NMS (Toàn bộ 8.400 hộp từ các thang đo P3, P4, P5)
        cand_scores = scores.cpu().numpy()
        cand_clss = clss.cpu().numpy().astype(int)
        cand_cxcywh = boxes_cxcywh.cpu().numpy()

        cand_x1 = np.clip(cand_cxcywh[:, 0] - cand_cxcywh[:, 2] / 2, 0, 640)
        cand_y1 = np.clip(cand_cxcywh[:, 1] - cand_cxcywh[:, 3] / 2, 0, 640)
        cand_x2 = np.clip(cand_cxcywh[:, 0] + cand_cxcywh[:, 2] / 2, 0, 640)
        cand_y2 = np.clip(cand_cxcywh[:, 1] + cand_cxcywh[:, 3] / 2, 0, 640)

        sort_order = np.argsort(-cand_scores)
        for idx_sorted in sort_order:
            c_idx = int(idx_sorted)
            c_id = int(cand_clss[idx_sorted])
            scale = "P3" if c_idx < 6400 else ("P4" if c_idx < 8000 else "P5")
            scale_name = "P3 (Vật thể nhỏ)" if c_idx < 6400 else ("P4 (Vật thể trung bình)" if c_idx < 8000 else "P5 (Vật thể lớn)")
            stride = 8 if c_idx < 6400 else (16 if c_idx < 8000 else 32)
            candidates.append({
                "box": [
                    round(float(cand_x1[idx_sorted]), 1),
                    round(float(cand_y1[idx_sorted]), 1),
                    round(float(cand_x2[idx_sorted]), 1),
                    round(float(cand_y2[idx_sorted]), 1)
                ],
                "confidence": round(float(cand_scores[idx_sorted]), 3),
                "class_id": c_id,
                "class_name": names.get(c_id, str(c_id)),
                "anchor_idx": c_idx,
                "scale": scale,
                "scale_name": scale_name,
                "stride": stride
            })

        # 2. Post-NMS Detections
        res_nms = nms.non_max_suppression(preds, conf_thres=conf_threshold, iou_thres=0.7, return_idxs=True)
        if len(res_nms) == 2 and len(res_nms[0]) > 0 and len(res_nms[0][0]) > 0 and len(res_nms[1][0]) > 0:
            det_boxes = res_nms[0][0].cpu().numpy()
            det_idxs = res_nms[1][0].cpu().numpy()
            for i in range(len(det_boxes)):
                c_id = int(det_boxes[i, 5])
                idx_val = int(det_idxs[i])
                scale = "P3" if idx_val < 6400 else ("P4" if idx_val < 8000 else "P5")
                scale_name = "P3 (Vật thể nhỏ)" if idx_val < 6400 else ("P4 (Vật thể trung bình)" if idx_val < 8000 else "P5 (Vật thể lớn)")
                stride = 8 if idx_val < 6400 else (16 if idx_val < 8000 else 32)
                detections.append({
                    "box": [round(float(x), 1) for x in det_boxes[i, :4]],
                    "confidence": round(float(det_boxes[i, 4]), 3),
                    "class_id": c_id,
                    "class_name": names.get(c_id, str(c_id)),
                    "anchor_idx": idx_val,
                    "scale": scale,
                    "scale_name": scale_name,
                    "stride": stride
                })

    elif isinstance(out, torch.Tensor) and out.shape[-1] == 6: # YOLO26 End-to-End
        is_nms_free = True
        boxes_np = out[0].cpu().numpy()
        mask = boxes_np[:, 4] >= conf_threshold
        survived = boxes_np[mask]
        for b in survived:
            c_id = int(b[5])
            w = b[2] - b[0]
            h = b[3] - b[1]
            diag = (w * h) ** 0.5
            scale = "P3" if diag < 40 else ("P4" if diag < 120 else "P5")
            scale_name = "P3 (Vật thể nhỏ)" if diag < 40 else ("P4 (Vật thể trung bình)" if diag < 120 else "P5 (Vật thể lớn)")
            stride = 8 if diag < 40 else (16 if diag < 120 else 32)
            detections.append({
                "box": [round(float(x), 1) for x in b[:4]],
                "confidence": round(float(b[4]), 3),
                "class_id": c_id,
                "class_name": names.get(c_id, str(c_id)),
                "anchor_idx": -1,
                "scale": scale,
                "scale_name": scale_name,
                "stride": stride
            })
        candidates = list(detections)

    nms_stats = {
        "is_nms_free": is_nms_free,
        "candidate_count": len(candidates),
        "final_count": len(detections),
        "suppressed_count": max(0, len(candidates) - len(detections)),
        "suppression_rate": f"{(1.0 - len(detections) / max(1, len(candidates))) * 100:.1f}%" if (len(candidates) > 0 and not is_nms_free) else "0%",
        "scale_distribution": {
            "P3": sum(1 for d in detections if d.get("scale") == "P3"),
            "P4": sum(1 for d in detections if d.get("scale") == "P4"),
            "P5": sum(1 for d in detections if d.get("scale") == "P5")
        }
    }

    return {
        "model_name": model_name,
        "inference_time_ms": round(infer_duration_ms, 2),
        "activations": activations,
        "layer_info_map": layer_info_map,
        "detections": detections,
        "candidates": candidates,
        "nms_stats": nms_stats,
        "preprocessed_rgb": img_rgb
    }
