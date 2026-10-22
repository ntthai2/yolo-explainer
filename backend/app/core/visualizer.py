"""
Tensor Visualizer Engine for YOLO Explainer.
Generates:
1. Mean and Max Activation Heatmaps (using Turbo / Viridis colormaps).
2. Alpha-blended Overlays (Original image + Heatmap).
3. Filter Bank (Top-16 most active channels ranked by L2 Energy).
4. Annotated Detection Image with bounding boxes.
Fast Base64 encoding for instant web transmission.
"""

import io
import base64
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import cv2
from PIL import Image


def array_to_base64_jpeg(img_rgb: np.ndarray, quality: int = 85) -> str:
    """Encodes an RGB/BGR numpy array to base64 JPEG data URL."""
    if img_rgb.ndim == 2:
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2BGR)
    else:
        # Expecting RGB, convert to BGR for cv2
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    success, buffer = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        return ""
    b64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalizes 2D float array to [0, 255] uint8."""
    min_val = np.min(arr)
    max_val = np.max(arr)
    if max_val - min_val > 1e-6:
        norm = (arr - min_val) / (max_val - min_val)
    else:
        norm = np.zeros_like(arr)
    return (norm * 255.0).astype(np.uint8)


def process_layer_activation(
    activation: torch.Tensor,
    layer_meta: Dict[str, Any],
    original_rgb: np.ndarray,
    top_k_channels: int = 16
) -> Dict[str, Any]:
    """
    Processes a single layer's 4D tensor [1, C, H, W] into:
    - mean_heatmap (base64)
    - overlay_image (base64)
    - top_channels (list of base64 thumbnails and energy metrics)
    - shape: [B, C, H, W]
    """
    if activation.dim() != 4:
        # Fallback if tensor shape is not 4D
        return {
            "id": layer_meta["id"],
            "name": layer_meta["name"],
            "shape": list(activation.shape),
            "role": layer_meta["role"],
            "stage": layer_meta["stage"],
            "type": layer_meta["type"],
            "badge": layer_meta.get("badge", ""),
            "receptive_field": layer_meta.get("receptive_field", ""),
            "mean_heatmap": None,
            "overlay": None,
            "top_channels": []
        }

    # Extract 3D tensor: [C, H, W]
    act_3d = activation[0].float().numpy()
    C, H, W = act_3d.shape
    preview_dim = 320

    # 1. Mean Heatmap across all channels (320x320 for fast transfer, crisp Retina 2x preview)
    mean_2d = np.mean(act_3d, axis=0)
    norm_mean = normalize_to_uint8(mean_2d)

    # Upsample to preview dimension using Bicubic
    heatmap_resized = cv2.resize(norm_mean, (preview_dim, preview_dim), interpolation=cv2.INTER_CUBIC)
    
    # Apply Turbo colormap
    colored_bgr = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_TURBO)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
    mean_heatmap_b64 = array_to_base64_jpeg(colored_rgb, quality=76)

    # 2. Alpha-blended Overlay (320x320)
    # Blend: 0.45 * original + 0.55 * heatmap
    orig_resized = cv2.resize(original_rgb, (preview_dim, preview_dim), interpolation=cv2.INTER_AREA)
    overlay_rgb = cv2.addWeighted(orig_resized, 0.45, colored_rgb, 0.55, 0)
    overlay_b64 = array_to_base64_jpeg(overlay_rgb, quality=76)

    # 3. Top Channels Ranking (L2 Energy)
    # Channel energy = sqrt(sum(x^2))
    energies = np.linalg.norm(act_3d.reshape(C, -1), axis=1) # shape: (C,)
    sorted_indices = np.argsort(energies)[::-1] # Descending

    top_count = min(top_k_channels, C)
    top_channels_list: List[Dict[str, Any]] = []

    for rank, ch_idx in enumerate(sorted_indices[:top_count]):
        ch_slice = act_3d[ch_idx]
        ch_norm = normalize_to_uint8(ch_slice)
        
        # Color each channel slice with Viridis for diversity/contrast
        ch_bgr = cv2.applyColorMap(ch_norm, cv2.COLORMAP_VIRIDIS)
        ch_rgb = cv2.cvtColor(ch_bgr, cv2.COLOR_BGR2RGB)
        
        # Resize thumbnail to 96x96 for crisp, fast UI display
        ch_thumb = cv2.resize(ch_rgb, (96, 96), interpolation=cv2.INTER_NEAREST)
        ch_b64 = array_to_base64_jpeg(ch_thumb, quality=72)

        top_channels_list.append({
            "channel_index": int(ch_idx),
            "rank": rank + 1,
            "energy": round(float(energies[ch_idx]), 2),
            "min_val": round(float(np.min(ch_slice)), 3),
            "max_val": round(float(np.max(ch_slice)), 3),
            "image": ch_b64
        })

    return {
        "id": layer_meta["id"],
        "name": layer_meta["name"],
        "shape": [1, C, H, W],
        "channel_count": C,
        "spatial_size": f"{W}x{H}",
        "role": layer_meta["role"],
        "stage": layer_meta["stage"],
        "type": layer_meta["type"],
        "badge": layer_meta.get("badge", ""),
        "receptive_field": layer_meta.get("receptive_field", ""),
        "mean_heatmap": mean_heatmap_b64,
        "overlay": overlay_b64,
        "top_channels": top_channels_list
    }


def render_detections_image(
    original_rgb: np.ndarray,
    detections: List[Dict[str, Any]]
) -> str:
    """
    Renders bounding boxes, class names, confidence and scale attribution badges (P3, P4, P5).
    """
    canvas = original_rgb.copy()
    
    PALETTE = [
        (59, 130, 246),  # Blue
        (16, 185, 129),  # Emerald
        (245, 158, 11),  # Amber
        (239, 68, 68),   # Red
        (168, 85, 247),  # Purple
        (236, 72, 153),  # Pink
        (20, 184, 166),  # Teal
        (249, 115, 22),  # Orange
    ]

    for det in detections:
        box = det["box"]
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cls_id = det["class_id"]
        cls_name = det["class_name"]
        conf = det["confidence"]
        scale = det.get("scale", "")
        
        color = PALETTE[cls_id % len(PALETTE)]

        # Draw smooth box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        # Draw label badge with scale attribution
        scale_tag = f" [{scale}]" if scale else ""
        label = f"{cls_name} {int(conf * 100)}%{scale_tag}"
        font_scale = 0.48
        font_thickness = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        
        badge_y1 = max(0, y1 - th - baseline - 5)
        badge_y2 = y1
        badge_x2 = min(canvas.shape[1], x1 + tw + 8)

        # Label background
        cv2.rectangle(canvas, (x1, badge_y1), (badge_x2, badge_y2), color, -1)
        # Label text (white)
        cv2.putText(
            canvas,
            label,
            (x1 + 4, badge_y2 - baseline - 1),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA
        )

    return array_to_base64_jpeg(canvas, quality=85)


def render_candidates_image(
    original_rgb: np.ndarray,
    candidates: List[Dict[str, Any]],
    detections: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Renders dense raw candidate bounding boxes before NMS with scale-coded glow
    and highlights final surviving detections.
    """
    canvas = original_rgb.copy()
    overlay = original_rgb.copy()
    
    # Scale colors for candidates: P3 (Small - Blue), P4 (Medium - Emerald), P5 (Large - Purple)
    SCALE_COLORS = {
        "P3": (59, 130, 246),
        "P4": (16, 185, 129),
        "P5": (168, 85, 247),
    }

    # Render top ~100 overlapping candidate boxes so clusters around objects are vividly visible
    top_candidates = candidates[:100]

    # Draw dense candidate boxes on overlay with thin 1px lines
    for cand in top_candidates:
        box = cand["box"]
        x1 = max(0, min(canvas.shape[1] - 1, int(box[0])))
        y1 = max(0, min(canvas.shape[0] - 1, int(box[1])))
        x2 = max(0, min(canvas.shape[1] - 1, int(box[2])))
        y2 = max(0, min(canvas.shape[0] - 1, int(box[3])))
        scale = cand.get("scale", "P4")
        color = SCALE_COLORS.get(scale, (59, 130, 246))
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

    # Blend overlay with canvas: alpha=0.75 ensures the thin overlapping candidate boxes are clearly distinct
    cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

    # Highlight surviving detections with solid outline
    if detections:
        for det in detections:
            box = det["box"]
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            scale = det.get("scale", "P5")
            color = SCALE_COLORS.get(scale, (239, 68, 68))
            
            # Thick border for survivor
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            
            label = f"{det['class_name']} {int(det['confidence'] * 100)}% ({scale})"
            font_scale = 0.45
            font_thickness = 1
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            badge_y1 = max(0, y1 - th - baseline - 4)
            badge_y2 = y1
            badge_x2 = min(canvas.shape[1], x1 + tw + 8)
            cv2.rectangle(canvas, (x1, badge_y1), (badge_x2, badge_y2), color, -1)
            cv2.putText(
                canvas,
                label,
                (x1 + 4, badge_y2 - baseline - 1),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness,
                cv2.LINE_AA
            )

    return array_to_base64_jpeg(canvas, quality=85)
