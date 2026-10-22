# YOLO Explainer

Ứng dụng web trực quan hóa kiến trúc mạng và feature map qua các thế hệ mô hình YOLOv8, YOLO11 và YOLO26.

---

## 1. Tính Năng Chính

- **So sánh 3 thế hệ kiến trúc YOLOv8n, YOLO11n và YOLO26n**:
  - **YOLOv8n**: Module C2f, Decoupled Head, hàm mất mát DFL, lọc trùng bằng thuật toán NMS.
  - **YOLO11n**: Module C3k2, cơ chế chú ý không gian C2PSA.
  - **YOLO26n**: Kiến trúc End-to-End Head loại bỏ bước NMS, đặt reg_max=1 để tính trực tiếp tọa độ hộp.
- **Sơ đồ luồng dữ liệu DAG**:
  - Hiển thị cấu trúc mạng theo ba phân vùng: Backbone, Neck và Head.
  - Thông tin kích thước tensor và bản đồ nhiệt kích hoạt trên từng khối.
- **Phân tích chi tiết layer**:
  - **Overlay Slider**: Thanh trượt so sánh trực tiếp giữa ảnh gốc và bản đồ nhiệt kích hoạt.
  - **Filter Bank Top 16**: Danh sách 16 kênh có chuẩn năng lượng L2 lớn nhất trong tensor.
  - **Kiến trúc và nguyên lý**: Giải thích chức năng và đặc điểm kỹ thuật của từng khối.
  - **Đầu ra phát hiện**: Bounding box, nhãn lớp và điểm tin cậy sau suy luận.
- **Giải thích Compound Scaling**:
  - Bảng thông số tỷ lệ phóng to mạng cho các variant n, s, m, l, x dựa trên hệ số độ sâu d và hệ số độ rộng w.

---

## 2. Hướng Dẫn Khởi Động

Khởi động máy chủ với cổng mặc định 8008:

```bash
cd /srv/nns/users/ntthai/yolo-explainer
./run.sh 8008
```

Mở trình duyệt và truy cập:
```
http://localhost:8008
```

---

## 3. Cấu Trúc Thư Mục

```
yolo-explainer/
├── backend/
│   ├── app/
│   │   ├── core/
│   │   │   ├── extractor.py        # Forward hook PyTorch trích xuất tensor và detection
│   │   │   ├── visualizer.py       # Xử lý bản đồ nhiệt, xếp hạng kênh L2 và vẽ bounding box
│   │   │   └── model_registry.py   # Cấu hình kiến trúc, metadata và quản lý model
│   │   ├── static/
│   │   │   └── samples/            # Ảnh mẫu đầu vào
│   │   └── main.py                 # FastAPI server và định tuyến tĩnh
│   ├── weights/                    # Trọng số mô hình .pt
│   └── requirements.txt
├── frontend/
│   ├── css/
│   │   └── style.css               # Giao diện người dùng
│   ├── js/
│   │   ├── app.js                  # Xử lý sự kiện và điều phối gọi API
│   │   ├── dag_renderer.js         # Vẽ sơ đồ DAG
│   │   └── inspector.js            # Xử lý thanh trượt so sánh và hiển thị chi tiết
│   └── index.html                  # Trang giao diện chính
├── run.sh                          # Script khởi chạy hệ thống
├── yolo_explainer_plan.md          # Kế hoạch kiến trúc ban đầu
└── README.md
```
