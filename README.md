# YOLO Explainer

Ứng dụng web trực quan hóa cấu trúc mạng nơ-ron và feature map của các mô hình YOLOv8, YOLO11 và YOLO26.

---

## 1. Giới thiệu

YOLO Explainer cung cấp giao diện tương tác hỗ trợ phân tích luồng xử lý bên trong của mô hình phát hiện vật thể YOLO, bao gồm các tầng trích xuất đặc trưng (Backbone), tầng tổng hợp đặc trưng đa tỷ lệ (Neck - PAN-FPN) và tầng dự đoán (Detect Head).

---

## 2. Đặc điểm kỹ thuật của các mô hình hỗ trợ

Hệ thống hỗ trợ phân tích ba biến thể Nano của họ mô hình YOLO:

- **YOLOv8n (2023)**:
  - Sử dụng module C2f trong Backbone và Neck để truyền luồng gradient qua các nhánh song song.
  - Sử dụng Decoupled Head, tách biệt nhánh phân loại lớp (classification) và nhánh định vị hộp (regression).
  - Hồi quy tọa độ bounding box bằng phương pháp DFL (Distribution Focal Loss) với `reg_max=16`.
  - Hậu xử lý kết quả dự đoán bằng thuật toán NMS (Non-Maximum Suppression) để loại bỏ các hộp chồng lấn.
- **YOLO11n (2024)**:
  - Thay thế module C2f bằng module C3k2 nhằm thay đổi cấu trúc tích chập.
  - Tích hợp khối chú ý không gian C2PSA (Cross Stage Partial with Spatial Attention) ở cuối Backbone để gán trọng số cho các vùng không gian.
  - Vẫn sử dụng cơ chế Decoupled Head, DFL và NMS tương tự YOLOv8.
- **YOLO26n (2026)**:
  - Sử dụng kiến trúc End-to-End Head được huấn luyện theo cơ chế gán nhãn đơn định (One-to-One Matching), không cần bước lọc trùng NMS ở giai đoạn suy luận.
  - Thiết lập `reg_max=1`, dự đoán trực tiếp giá trị tọa độ bounding box thay vì dự đoán phân phối xác suất DFL.
  - Thời gian suy luận ở tầng Head có tính chất cố định hơn do không phụ thuộc vào số lượng hộp ứng viên đưa vào NMS.

### Bảng so sánh thông số kiến trúc

| Thuộc tính | YOLOv8n | YOLO11n | YOLO26n |
| :--- | :--- | :--- | :--- |
| **Khối đặc trưng chính** | C2f | C3k2 | C3k2 |
| **Cơ chế Attention** | Không | C2PSA | C2PSA |
| **Kiến trúc Head** | Decoupled Head | Decoupled Head | End-to-End NMS-Free Head |
| **Cơ chế hồi quy tọa độ** | DFL (`reg_max=16`) | DFL (`reg_max=16`) | Trực tiếp (`reg_max=1`) |
| **Hậu xử lý NMS** | Có | Có | Không |

---

## 3. Các chức năng chính

- **Sơ đồ luồng xử lý (DAG Pipeline Flowchart)**: Hiển thị các khối mạng theo ba phân vùng: Backbone, Neck và Head; cung cấp kích thước tensor đầu ra `[C, H, W]` và bản đồ kích hoạt tương ứng tại từng tầng quan sát.
- **Thanh trượt so sánh (Overlay Slider)**: Cho phép kéo trượt để chồng bản đồ nhiệt kích hoạt (heatmap) lên ảnh gốc đầu vào.
- **Phân tích kênh (Filter Bank)**: Xếp hạng và hiển thị các kênh có độ lớn chuẩn L2 lớn nhất trong tensor kích hoạt của tầng được chọn.
- **So sánh ứng viên và kết quả phát hiện**:
  - Đối với YOLOv8 và YOLO11: Hiển thị tập hợp các hộp ứng viên trước bước NMS và tập hợp các hộp sau khi lọc NMS, kèm tỷ lệ loại bỏ (suppression rate).
  - Đối với YOLO26: Hiển thị trực tiếp các phát hiện từ đầu ra của mạng.
- **Bảng thông số Compound Scaling**: Cung cấp thông số độ sâu ($d$), độ rộng ($w$), số lượng tham số và FLOPs lý thuyết của các biến thể quy mô từ $n$ đến $x$.

---

## 4. Yêu cầu hệ thống và cài đặt

### Yêu cầu môi trường
- Hệ điều hành: Linux / macOS / Windows
- Python >= 3.9
- PyTorch >= 2.0 (hỗ trợ cả CPU và GPU qua CUDA)

### Các bước cài đặt

1. Sao chép mã nguồn:
   ```bash
   git clone https://github.com/ntthai2/yolo-explainer.git
   cd yolo-explainer
   ```

2. Tạo và kích hoạt môi trường ảo:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Cài đặt các gói phụ thuộc:
   ```bash
   pip install -r backend/requirements.txt
   ```

---

## 5. Quản lý trọng số mô hình (Weights)

Các tệp trọng số mô hình định dạng `.pt` được loại trừ trong tệp `.gitignore` để không lưu trữ nhị phân dung lượng lớn trên Git.

- Vị trí tệp trọng số: Đặt trong thư mục `weights/` hoặc `backend/weights/`:
  - `yolov8n.pt`
  - `yolo11n.pt`
  - `yolo26n.pt`
- Với các mô hình chuẩn (`yolov8n.pt`, `yolo11n.pt`), thư viện Ultralytics sẽ tự động tải về nếu tệp chưa tồn tại cục bộ khi chạy lần đầu.

---

## 6. Hướng dẫn vận hành

### Khởi động bằng script `run.sh`
Script tự động đặt đường dẫn dự án vào `PYTHONPATH` và hỗ trợ chuyển tiếp cổng nếu cấu hình SSH tunnel:

```bash
chmod +x run.sh
./run.sh 8008
```

### Khởi động trực tiếp bằng Uvicorn
```bash
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8008
```

Mở trình duyệt web và truy cập địa chỉ:
```
http://localhost:8008
```

---

## 7. Giao diện lập trình ứng dụng (REST API)

FastAPI backend cung cấp các endpoint sau:

- `GET /api/health`: Trả về trạng thái hoạt động của hệ thống, thiết bị tính toán (`cpu` hoặc `cuda:0`) và danh sách mô hình được cấu hình.
- `GET /api/models`: Cung cấp danh sách mô hình, metadata kiến trúc và danh sách tầng quan sát.
- `GET /api/samples`: Trả về danh sách ảnh mẫu có sẵn trên hệ thống.
- `POST /api/inspect`: Nhận tham số `model_name` cùng ảnh tải lên (hoặc `sample_id`), trả về tensor kích hoạt, bản đồ nhiệt, danh sách kênh hàng đầu và kết quả phát hiện.

---

## 8. Cấu trúc thư mục

```
yolo-explainer/
├── backend/
│   ├── app/
│   │   ├── core/
│   │   │   ├── extractor.py        # Forward hook trích xuất tensor kích hoạt và dự đoán
│   │   │   ├── model_registry.py   # Định nghĩa cấu hình kiến trúc, metadata và hệ số scaling
│   │   │   └── visualizer.py       # Tính toán bản đồ nhiệt, chuẩn L2 kênh và vẽ bounding box
│   │   ├── static/
│   │   │   └── samples/            # Tệp ảnh mẫu phục vụ thử nghiệm
│   │   └── main.py                 # FastAPI server và định tuyến API
│   ├── weights/                    # Thư mục lưu trữ trọng số mô hình (.pt)
│   └── requirements.txt            # Danh sách thư viện Python phụ thuộc
├── frontend/
│   ├── css/
│   │   └── style.css               # Định dạng giao diện người dùng
│   ├── js/
│   │   ├── app.js                  # Điều phối sự kiện và giao tiếp API
│   │   ├── arch_diagram.js         # Vẽ sơ đồ cấu trúc kiến trúc
│   │   ├── dag_renderer.js         # Vẽ sơ đồ luồng DAG của các tầng mạng
│   │   └── inspector.js            # Xử lý thanh trượt đối chiếu và hiển thị chi tiết
│   ├── vendor/
│   │   └── katex/                  # Thư viện hiển thị công thức toán học
│   └── index.html                  # Giao diện chính của ứng dụng web
├── run.sh                          # Script khởi động hệ thống
├── .gitignore                      # Cấu hình các tệp không theo dõi trên Git
└── README.md                       # Tài liệu hướng dẫn kỹ thuật
```

---

## 9. Giấy phép

Dự án được phân phối theo giấy phép MIT. Chi tiết xem tại tệp `LICENSE`.
