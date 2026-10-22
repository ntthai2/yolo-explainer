"""
Model Registry for YOLO Explainer.
Manages metadata, layer configurations, and model instances for YOLOv8n, YOLO11n, and YOLO26n.
"""

import os
from typing import Dict, Any, List
import torch
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "yolov8n": {
        "id": "yolov8n",
        "name": "YOLOv8n",
        "title": "YOLOv8 Nano",
        "year": "2023",
        "family": "v8",
        "weight_file": os.path.join(WEIGHTS_DIR, "yolov8n.pt"),
        "key_features": [
            "Module C2f tối ưu luồng gradient",
            "Decoupled Head tách riêng nhánh phân loại và định vị",
            "Hồi quy bounding box bằng DFL",
            "Gán nhãn Task-Aligned Assigner và lọc trùng bằng NMS"
        ],
        "compound_scaling": {
            "n": {"depth": 0.33, "width": 0.25, "max_channels": 1024, "params": "3.2M", "flops": "8.7G", "desc": "Phiên bản nhỏ nhất, tối ưu tốc độ và dung lượng bộ nhớ"},
            "s": {"depth": 0.33, "width": 0.50, "max_channels": 1024, "params": "11.2M", "flops": "28.6G", "desc": "Cân bằng giữa tốc độ và độ chính xác"},
            "m": {"depth": 0.67, "width": 0.75, "max_channels": 768, "params": "25.9M", "flops": "78.9G", "desc": "Phù hợp máy chủ và phần cứng tầm trung"},
            "l": {"depth": 1.00, "width": 1.00, "max_channels": 512, "params": "43.7M", "flops": "165.2G", "desc": "Độ chính xác cao hơn, yêu cầu tài nguyên tính toán lớn hơn"},
            "x": {"depth": 1.00, "width": 1.25, "max_channels": 512, "params": "68.2M", "flops": "257.8G", "desc": "Độ chính xác cao nhất trong nhóm kiến trúc v8"}
        },
        "monitored_layers": [
            {
                "id": "layer_0_stem",
                "layer_idx": 0,
                "name": "Stage 0: Stem Conv",
                "stage": "backbone",
                "type": "Conv",
                "role": "Trích xuất đặc trưng biên và họa tiết cơ bản từ ảnh đầu vào ở stride 2",
                "receptive_field": "3x3",
                "badge": "Stem Stride 2"
            },
            {
                "id": "layer_2_p2",
                "layer_idx": 2,
                "name": "Stage 2: C2f P2",
                "stage": "backbone",
                "type": "C2f",
                "role": "Đặc trưng hình học mức thấp ở stride 4",
                "receptive_field": "7x7",
                "badge": "P2 Stride 4"
            },
            {
                "id": "layer_4_p3",
                "layer_idx": 4,
                "name": "Stage 4: C2f P3",
                "stage": "backbone",
                "type": "C2f",
                "role": "Đặc trưng cho vật thể nhỏ ở stride 8",
                "receptive_field": "15x15",
                "badge": "P3 Stride 8"
            },
            {
                "id": "layer_6_p4",
                "layer_idx": 6,
                "name": "Stage 6: C2f P4",
                "stage": "backbone",
                "type": "C2f",
                "role": "Đặc trưng cho vật thể trung bình ở stride 16",
                "receptive_field": "31x31",
                "badge": "P4 Stride 16"
            },
            {
                "id": "layer_8_p5",
                "layer_idx": 8,
                "name": "Stage 8: C2f P5",
                "stage": "backbone",
                "type": "C2f",
                "role": "Đặc trưng ngữ nghĩa mức cao cho vật thể lớn ở stride 32",
                "receptive_field": "63x63",
                "badge": "P5 Stride 32"
            },
            {
                "id": "layer_9_sppf",
                "layer_idx": 9,
                "name": "Stage 9: SPPF",
                "stage": "backbone",
                "type": "SPPF",
                "role": "Spatial Pyramid Pooling Fast tổng hợp ngữ cảnh đa thang đo",
                "receptive_field": "Multi-scale",
                "badge": "SPPF Pooling"
            },
            {
                "id": "layer_12_neck_p4",
                "layer_idx": 12,
                "name": "Neck: Top-Down Fusion P4",
                "stage": "neck",
                "type": "C2f",
                "role": "FPN Path truyền đặc trưng ngữ nghĩa từ tầng P5 xuống tầng P4",
                "receptive_field": "Multi-scale",
                "badge": "FPN P4"
            },
            {
                "id": "layer_15_neck_p3",
                "layer_idx": 15,
                "name": "Neck: Top-Down Fusion P3",
                "stage": "neck",
                "type": "C2f",
                "role": "FPN Path truyền đặc trưng ngữ nghĩa xuống tầng P3",
                "receptive_field": "Multi-scale",
                "badge": "FPN P3"
            },
            {
                "id": "layer_18_neck_pan_p4",
                "layer_idx": 18,
                "name": "Neck: Bottom-Up Path PAN P4",
                "stage": "neck",
                "type": "C2f",
                "role": "PAN Path truyền thông tin vị trí ngược từ tầng dưới lên tầng P4",
                "receptive_field": "Multi-scale",
                "badge": "PAN P4"
            },
            {
                "id": "layer_21_neck_pan_p5",
                "layer_idx": 21,
                "name": "Neck: Bottom-Up Path PAN P5",
                "stage": "neck",
                "type": "C2f",
                "role": "PAN Path tổng hợp thông tin trước khi chuyển vào Detect Head",
                "receptive_field": "Multi-scale",
                "badge": "PAN P5"
            },
            {
                "id": "layer_22_head",
                "layer_idx": 22,
                "name": "Stage 22: Decoupled Detect Head",
                "stage": "head",
                "type": "Detect",
                "role": "Phân nhánh dự đoán hộp với DFL và phân loại lớp rồi lọc trùng bằng NMS",
                "receptive_field": "P3 P4 P5 Outputs",
                "badge": "Decoupled + NMS"
            }
        ]
    },
    "yolo11n": {
        "id": "yolo11n",
        "name": "YOLO11n",
        "title": "YOLO11 Nano",
        "year": "2024",
        "family": "11",
        "weight_file": os.path.join(WEIGHTS_DIR, "yolo11n.pt"),
        "key_features": [
            "Module C3k2 giảm phép tính và tham số",
            "Cơ chế chú ý không gian C2PSA",
            "Tối ưu cấu trúc luồng đặc trưng tại Backbone và Neck",
            "Duy trì kích thước nhỏ với độ chính xác cao"
        ],
        "compound_scaling": {
            "n": {"depth": 0.50, "width": 0.25, "max_channels": 1024, "params": "2.6M", "flops": "6.5G", "desc": "Giảm tham số so với v8n, phù hợp thiết bị nhẹ"},
            "s": {"depth": 0.50, "width": 0.50, "max_channels": 1024, "params": "9.4M", "flops": "21.5G", "desc": "Tối ưu cho phần cứng xử lý biên"},
            "m": {"depth": 0.50, "width": 1.00, "max_channels": 512, "params": "20.1M", "flops": "68.0G", "desc": "Cân bằng giữa tốc độ và độ chính xác"},
            "l": {"depth": 1.00, "width": 1.00, "max_channels": 512, "params": "25.3M", "flops": "86.9G", "desc": "Độ chính xác cao, số lượng tham số thấp hơn v8l"},
            "x": {"depth": 1.00, "width": 1.50, "max_channels": 512, "params": "56.9M", "flops": "194.9G", "desc": "Độ chính xác cao nhất trong nhóm kiến trúc 11"}
        },
        "monitored_layers": [
            {
                "id": "layer_0_stem",
                "layer_idx": 0,
                "name": "Stage 0: Stem Conv",
                "stage": "backbone",
                "type": "Conv",
                "role": "Trích xuất đặc trưng biên và họa tiết cơ bản ở stride 2",
                "receptive_field": "3x3",
                "badge": "Stem Stride 2"
            },
            {
                "id": "layer_2_p2",
                "layer_idx": 2,
                "name": "Stage 2: C3k2 P2",
                "stage": "backbone",
                "type": "C3k2",
                "role": "Trích xuất đặc trưng hình học mức thấp ở stride 4",
                "receptive_field": "7x7",
                "badge": "C3k2 Stride 4"
            },
            {
                "id": "layer_4_p3",
                "layer_idx": 4,
                "name": "Stage 4: C3k2 P3",
                "stage": "backbone",
                "type": "C3k2",
                "role": "Đặc trưng cho vật thể nhỏ ở stride 8",
                "receptive_field": "15x15",
                "badge": "C3k2 Stride 8"
            },
            {
                "id": "layer_6_p4",
                "layer_idx": 6,
                "name": "Stage 6: C3k2 P4",
                "stage": "backbone",
                "type": "C3k2",
                "role": "Đặc trưng cho vật thể trung bình ở stride 16",
                "receptive_field": "31x31",
                "badge": "C3k2 Stride 16"
            },
            {
                "id": "layer_8_p5",
                "layer_idx": 8,
                "name": "Stage 8: C3k2 P5",
                "stage": "backbone",
                "type": "C3k2",
                "role": "Đặc trưng ngữ nghĩa mức cao cho vật thể lớn ở stride 32",
                "receptive_field": "63x63",
                "badge": "C3k2 Stride 32"
            },
            {
                "id": "layer_9_sppf",
                "layer_idx": 9,
                "name": "Stage 9: SPPF",
                "stage": "backbone",
                "type": "SPPF",
                "role": "Spatial Pyramid Pooling Fast tổng hợp ngữ cảnh đa thang đo",
                "receptive_field": "Multi-scale",
                "badge": "SPPF Pooling"
            },
            {
                "id": "layer_10_c2psa",
                "layer_idx": 10,
                "name": "Stage 10: C2PSA Attention",
                "stage": "backbone",
                "type": "C2PSA",
                "role": "Điều chỉnh trọng số không gian cho vùng đặc trưng mục tiêu",
                "receptive_field": "Self-Attention",
                "badge": "Attention C2PSA"
            },
            {
                "id": "layer_13_neck_p4",
                "layer_idx": 13,
                "name": "Neck: Top-Down C3k2 P4",
                "stage": "neck",
                "type": "C3k2",
                "role": "FPN Path truyền đặc trưng ngữ nghĩa từ tầng P5 xuống tầng P4",
                "receptive_field": "Multi-scale",
                "badge": "FPN C3k2"
            },
            {
                "id": "layer_16_neck_p3",
                "layer_idx": 16,
                "name": "Neck: Top-Down C3k2 P3",
                "stage": "neck",
                "type": "C3k2",
                "role": "FPN Path truyền đặc trưng ngữ nghĩa xuống tầng P3",
                "receptive_field": "Multi-scale",
                "badge": "FPN P3"
            },
            {
                "id": "layer_19_neck_pan_p4",
                "layer_idx": 19,
                "name": "Neck: Bottom-Up C3k2 PAN P4",
                "stage": "neck",
                "type": "C3k2",
                "role": "PAN Path truyền thông tin vị trí ngược từ tầng dưới lên tầng P4",
                "receptive_field": "Multi-scale",
                "badge": "PAN P4"
            },
            {
                "id": "layer_22_neck_pan_p5",
                "layer_idx": 22,
                "name": "Neck: Bottom-Up C3k2 PAN P5",
                "stage": "neck",
                "type": "C3k2",
                "role": "PAN Path tổng hợp thông tin trước khi chuyển vào Detect Head",
                "receptive_field": "Multi-scale",
                "badge": "PAN P5"
            },
            {
                "id": "layer_23_head",
                "layer_idx": 23,
                "name": "Stage 23: Detect Head",
                "stage": "head",
                "type": "Detect",
                "role": "Dự đoán lớp và vị trí đối tượng trên các thang đo P3 P4 P5",
                "receptive_field": "P3 P4 P5 Outputs",
                "badge": "Detect Head"
            }
        ]
    },
    "yolo26n": {
        "id": "yolo26n",
        "name": "YOLO26n",
        "title": "YOLO26 Nano",
        "year": "2026",
        "family": "26",
        "weight_file": os.path.join(WEIGHTS_DIR, "yolo26n.pt"),
        "key_features": [
            "Kiến trúc End-to-End không yêu cầu thuật toán NMS hậu xử lý",
            "Thiết lập reg_max=1 loại bỏ DFL và tính tọa độ trực tiếp",
            "Tích hợp khối chú ý C2PSA và khối tích chập C3k2",
            "Độ trễ suy luận ổn định không phụ thuộc số lượng hộp phát hiện"
        ],
        "compound_scaling": {
            "n": {"depth": 0.50, "width": 0.25, "max_channels": 1024, "params": "2.4M", "flops": "6.1G", "desc": "Tối ưu độ trễ, phù hợp suy luận trực tiếp không qua NMS"},
            "s": {"depth": 0.50, "width": 0.50, "max_channels": 1024, "params": "8.9M", "flops": "20.8G", "desc": "Cân bằng tốc độ và độ chính xác cho thiết bị biên"},
            "m": {"depth": 0.50, "width": 1.00, "max_channels": 512, "params": "19.5M", "flops": "65.4G", "desc": "Phù hợp máy chủ và phần cứng tầm trung"},
            "l": {"depth": 1.00, "width": 1.00, "max_channels": 512, "params": "24.8M", "flops": "84.2G", "desc": "Độ chính xác cao hơn trên tập dữ liệu phức tạp"},
            "x": {"depth": 1.00, "width": 1.50, "max_channels": 512, "params": "55.4M", "flops": "189.2G", "desc": "Độ chính xác cao nhất trong nhóm kiến trúc 26"}
        },
        "monitored_layers": [
            {
                "id": "layer_0_stem",
                "layer_idx": 0,
                "name": "Stage 0: Stem Conv",
                "stage": "backbone",
                "type": "Conv",
                "role": "Trích xuất đặc trưng biên và họa tiết cơ bản ở stride 2",
                "receptive_field": "3x3",
                "badge": "Stem Stride 2"
            },
            {
                "id": "layer_2_p2",
                "layer_idx": 2,
                "name": "Stage 2: C3k2 P2",
                "stage": "backbone",
                "type": "C3k2",
                "role": "Đặc trưng hình học mức thấp ở stride 4",
                "receptive_field": "7x7",
                "badge": "C3k2 Stride 4"
            },
            {
                "id": "layer_4_p3",
                "layer_idx": 4,
                "name": "Stage 4: C3k2 P3",
                "stage": "backbone",
                "type": "C3k2",
                "role": "Đặc trưng cho vật thể nhỏ ở stride 8",
                "receptive_field": "15x15",
                "badge": "C3k2 Stride 8"
            },
            {
                "id": "layer_6_p4",
                "layer_idx": 6,
                "name": "Stage 6: C3k2 P4",
                "stage": "backbone",
                "type": "C3k2",
                "role": "Đặc trưng cho vật thể trung bình ở stride 16",
                "receptive_field": "31x31",
                "badge": "C3k2 Stride 16"
            },
            {
                "id": "layer_8_p5",
                "layer_idx": 8,
                "name": "Stage 8: C3k2 P5",
                "stage": "backbone",
                "type": "C3k2",
                "role": "Đặc trưng ngữ nghĩa mức cao cho vật thể lớn ở stride 32",
                "receptive_field": "63x63",
                "badge": "C3k2 Stride 32"
            },
            {
                "id": "layer_9_sppf",
                "layer_idx": 9,
                "name": "Stage 9: SPPF",
                "stage": "backbone",
                "type": "SPPF",
                "role": "Spatial Pyramid Pooling Fast tổng hợp ngữ cảnh đa thang đo",
                "receptive_field": "Multi-scale",
                "badge": "SPPF Pooling"
            },
            {
                "id": "layer_10_c2psa",
                "layer_idx": 10,
                "name": "Stage 10: C2PSA Attention",
                "stage": "backbone",
                "type": "C2PSA",
                "role": "Điều chỉnh trọng số không gian cho vùng đặc trưng mục tiêu",
                "receptive_field": "Self-Attention",
                "badge": "Attention C2PSA"
            },
            {
                "id": "layer_13_neck_p4",
                "layer_idx": 13,
                "name": "Neck: Top-Down C3k2 P4",
                "stage": "neck",
                "type": "C3k2",
                "role": "FPN Path truyền đặc trưng ngữ nghĩa từ tầng P5 xuống tầng P4",
                "receptive_field": "Multi-scale",
                "badge": "FPN C3k2"
            },
            {
                "id": "layer_16_neck_p3",
                "layer_idx": 16,
                "name": "Neck: Top-Down C3k2 P3",
                "stage": "neck",
                "type": "C3k2",
                "role": "FPN Path truyền đặc trưng ngữ nghĩa xuống tầng P3",
                "receptive_field": "Multi-scale",
                "badge": "FPN P3"
            },
            {
                "id": "layer_19_neck_pan_p4",
                "layer_idx": 19,
                "name": "Neck: Bottom-Up C3k2 PAN P4",
                "stage": "neck",
                "type": "C3k2",
                "role": "PAN Path truyền thông tin vị trí ngược từ tầng dưới lên tầng P4",
                "receptive_field": "Multi-scale",
                "badge": "PAN P4"
            },
            {
                "id": "layer_22_neck_pan_p5",
                "layer_idx": 22,
                "name": "Neck: Bottom-Up C3k2 PAN P5",
                "stage": "neck",
                "type": "C3k2",
                "role": "PAN Path tổng hợp thông tin trước khi chuyển vào Detect Head",
                "receptive_field": "Multi-scale",
                "badge": "PAN P5"
            },
            {
                "id": "layer_23_head",
                "layer_idx": 23,
                "name": "Stage 23: End-to-End NMS-Free Head",
                "stage": "head",
                "type": "Detect",
                "role": "Phân bổ One-to-One Matching trong huấn luyện và loại bỏ bước lọc trùng NMS",
                "receptive_field": "P3 P4 P5 Outputs",
                "badge": "NMS-Free End2End"
            }
        ]
    }
}


class ModelRegistry:
    """Manages loading and caching of YOLO models."""
    def __init__(self):
        self._loaded_models: Dict[str, YOLO] = {}
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def get_model(self, model_name: str) -> YOLO:
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Model {model_name} không được hỗ trợ. Danh sách hỗ trợ: {list(MODEL_CONFIGS.keys())}")
        
        if model_name not in self._loaded_models:
            cfg = MODEL_CONFIGS[model_name]
            weight_path = cfg["weight_file"]
            if not os.path.exists(weight_path):
                weight_path = f"{model_name}.pt"
            yolo = YOLO(weight_path)
            yolo.model.to(self.device).eval()
            self._loaded_models[model_name] = yolo
            
        return self._loaded_models[model_name]

    def get_metadata(self, model_name: str) -> Dict[str, Any]:
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Model {model_name} không tồn tại.")
        cfg = MODEL_CONFIGS[model_name]
        return {
            "id": cfg["id"],
            "name": cfg["name"],
            "title": cfg["title"],
            "year": cfg["year"],
            "family": cfg["family"],
            "key_features": cfg["key_features"],
            "compound_scaling": cfg["compound_scaling"],
            "monitored_layers": cfg["monitored_layers"],
            "device": self.device
        }

    def list_all_models(self) -> List[Dict[str, Any]]:
        return [self.get_metadata(m) for m in MODEL_CONFIGS.keys()]


# Global singleton
registry = ModelRegistry()
