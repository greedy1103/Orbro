# Hệ thống phát hiện lửa và khói

Hệ thống này sử dụng YOLOv8 để phát hiện lửa và khói trong video hoặc stream camera. Hỗ trợ cả xử lý trên CPU/GPU thông thường và triển khai trên chip Hailo.

## Cấu trúc thư mục

```
Fire and Smoke/
├── data.yaml                # Cấu hình dữ liệu cho quá trình huấn luyện
├── detect.py               # Script phát hiện lửa và khói với YOLOv8
├── hailo_detect.py         # Script phát hiện lửa và khói với chip Hailo
├── train.py                # Script huấn luyện mô hình YOLOv8
├── convert_to_hailo.py     # Script chuyển đổi mô hình sang định dạng Hailo
├── requirements.txt        # Các thư viện cần thiết
└── README.md               # Hướng dẫn này
```

## Cài đặt

### Bước 1: Cài đặt các thư viện cần thiết

```bash
pip install -r requirements.txt
```

### Bước 2: Tải mô hình YOLOv8 hoặc huấn luyện mô hình mới

#### Tùy chọn A: Tải mô hình có sẵn
Tải mô hình YOLOv8n hoặc YOLOv8s và sử dụng trực tiếp:

```bash
wget https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov8n.pt
```

#### Tùy chọn B: Huấn luyện mô hình mới

Chuẩn bị dữ liệu theo định dạng YOLOv8:
- Tạo thư mục `datasets/fire_smoke/`
- Bên trong, tạo các thư mục `images/train/`, `images/val/`, và `images/test/`
- Thêm hình ảnh và file nhãn tương ứng

```bash
python train.py --data data.yaml --epochs 100 --batch 16 --imgsz 640
```

## Sử dụng

### Phát hiện lửa và khói với YOLOv8 (trên PC hoặc Raspberry Pi không có chip Hailo)

```bash
# Sử dụng webcam
python detect.py --view-img

# Sử dụng video
python detect.py --source test_video.mp4 --view-img --save

# Sử dụng RTSP stream
python detect.py --source "rtsp://username:password@ip:port/path" --view-img
```

### Chuyển đổi mô hình sang định dạng Hailo HEF

```bash
python convert_to_hailo.py --input runs/detect/fire_smoke_detector/weights/best.pt
```

### Phát hiện lửa và khói với chip Hailo trên Raspberry Pi

```bash
# Sử dụng webcam
python hailo_detect.py --model fire_smoke_detector.hef --view-img

# Sử dụng video
python hailo_detect.py --model fire_smoke_detector.hef --source test_video.mp4 --view-img --save

# Sử dụng RTSP stream
python hailo_detect.py --model fire_smoke_detector.hef --source "rtsp://username:password@ip:port/path" --view-img
```

## Lưu ý quan trọng

### Cài đặt Hailo SDK

Để sử dụng chip Hailo trên Raspberry Pi, bạn cần cài đặt Hailo SDK:

1. Tải Hailo SDK từ trang chủ của Hailo
2. Cài đặt SDK theo hướng dẫn của Hailo
3. Kết nối chip Hailo với Raspberry Pi

### Điều chỉnh hàm xử lý đầu ra

Hàm `postprocess_output` trong file `hailo_detect.py` cần được điều chỉnh dựa trên định dạng đầu ra cụ thể của mô hình Hailo của bạn.

### Tối ưu hiệu suất

- Giảm kích thước đầu vào nếu FPS quá thấp
- Điều chỉnh ngưỡng tin cậy để cân bằng giữa độ chính xác và hiệu suất
- Sử dụng định dạng H.264 cho RTSP để giảm băng thông và độ trễ

## Hiệu suất

- **Raspberry Pi 4 (không có Hailo)**: Khoảng 2-4 FPS
- **Raspberry Pi 4 + chip Hailo**: Khoảng 15-25 FPS
- **PC/Laptop với GPU**: Khoảng 30-60 FPS (tùy vào GPU) 