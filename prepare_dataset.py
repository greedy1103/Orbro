import os
import shutil
import random
from pathlib import Path
import json

def setup_kaggle_api():
    """Hướng dẫn cài đặt Kaggle API"""
    print("=== Cài đặt Kaggle API ===")
    print("1. Cài đặt kaggle package:")
    print("   pip install kaggle")
    print("\n2. Tạo Kaggle API token:")
    print("   - Đăng nhập vào Kaggle.com")
    print("   - Vào Account > Create New API Token")
    print("   - Tải file kaggle.json về")
    print("   - Đặt file tại: ~/.kaggle/kaggle.json (Linux/Mac) hoặc C:\\Users\\<username>\\.kaggle\\kaggle.json (Windows)")
    print("\n3. Chạy lại script này sau khi hoàn thành các bước trên.")

def download_dataset():
    """Tải dataset từ Kaggle"""
    dataset_name = "dataclusterlabs/fire-and-smoke-dataset"
    download_path = "datasets/raw"
    
    try:
        import kaggle
        print(f"Đang tải dataset: {dataset_name}")
        
        # Tạo thư mục nếu chưa có
        os.makedirs(download_path, exist_ok=True)
        
        # Tải dataset
        kaggle.api.dataset_download_files(dataset_name, path=download_path, unzip=True)
        print(f"Dataset đã được tải về thư mục: {download_path}")
        return True
        
    except ImportError:
        print("Lỗi: Chưa cài đặt kaggle package")
        print("Chạy: pip install kaggle")
        return False
    except Exception as e:
        print(f"Lỗi khi tải dataset: {str(e)}")
        print("Vui lòng kiểm tra cấu hình Kaggle API")
        return False

def convert_to_yolo_format():
    """Chuyển đổi dataset sang định dạng YOLO"""
    raw_path = Path("datasets/raw")
    output_path = Path("datasets/fire_smoke")
    
    # Tạo cấu trúc thư mục YOLO
    for split in ['train', 'val', 'test']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    print("Đang chuyển đổi dataset sang định dạng YOLO...")
    
    # Tìm tất cả file ảnh trong dataset
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    all_images = []
    
    for ext in image_extensions:
        all_images.extend(list(raw_path.rglob(f"*{ext}")))
        all_images.extend(list(raw_path.rglob(f"*{ext.upper()}")))
    
    print(f"Tìm thấy {len(all_images)} ảnh")
    
    # Phân chia dataset: 70% train, 20% val, 10% test
    random.shuffle(all_images)
    train_split = int(0.7 * len(all_images))
    val_split = int(0.9 * len(all_images))
    
    splits = {
        'train': all_images[:train_split],
        'val': all_images[train_split:val_split],
        'test': all_images[val_split:]
    }
    
    # Xử lý từng split
    for split_name, images in splits.items():
        print(f"Xử lý {split_name}: {len(images)} ảnh")
        
        for img_path in images:
            # Sao chép ảnh
            img_name = img_path.name
            dst_img = output_path / 'images' / split_name / img_name
            shutil.copy2(img_path, dst_img)
            
            # Tạo file label (giả định - cần điều chỉnh theo dataset thực tế)
            label_name = img_path.stem + '.txt'
            dst_label = output_path / 'labels' / split_name / label_name
            
            # Xác định class dựa trên tên file hoặc thư mục
            class_id = determine_class(img_path)
            
            # Tạo label giả định (bounding box toàn ảnh)
            # Trong thực tế, bạn cần có annotation thực sự
            with open(dst_label, 'w') as f:
                f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")  # class_id x_center y_center width height (normalized)
    
    print("Hoàn thành chuyển đổi dataset!")
    print(f"Dataset YOLO đã được lưu tại: {output_path}")

def determine_class(img_path):
    """Xác định class dựa trên tên file hoặc đường dẫn"""
    path_str = str(img_path).lower()
    
    if 'fire' in path_str:
        return 1  # fire
    elif 'smoke' in path_str:
        return 0  # smoke
    else:
        # Mặc định là smoke nếu không xác định được
        return 0

def update_data_yaml():
    """Cập nhật file data.yaml với đường dẫn dataset mới"""
    yaml_content = """path: datasets/fire_smoke  # đường dẫn đến thư mục gốc dataset
train: images/train  # đường dẫn đến thư mục ảnh huấn luyện
val: images/val  # đường dẫn đến thư mục ảnh xác thực 
test: images/test  # đường dẫn đến thư mục ảnh kiểm tra

# Các lớp (classes)
nc: 2  # số lượng lớp
names: ['smoke', 'fire']  # tên các lớp

# Thống kê dataset
# train: được tự động tính khi chạy
# val: được tự động tính khi chạy
"""
    
    with open('data.yaml', 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print("Đã cập nhật file data.yaml")

def create_better_labels():
    """Tạo script để cải thiện labels (cần chạy riêng nếu có annotation tools)"""
    script_content = '''
# Script này giúp tạo labels tốt hơn
# Bạn có thể sử dụng các công cụ như:
# 1. LabelImg: https://github.com/heartexlabs/labelImg
# 2. CVAT: https://github.com/opencv/cvat
# 3. Roboflow: https://roboflow.com

import os
from pathlib import Path

def create_empty_labels():
    """Tạo file label trống cho những ảnh chưa có annotation"""
    dataset_path = Path("datasets/fire_smoke")
    
    for split in ['train', 'val', 'test']:
        images_dir = dataset_path / 'images' / split
        labels_dir = dataset_path / 'labels' / split
        
        for img_file in images_dir.glob('*'):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                label_file = labels_dir / (img_file.stem + '.txt')
                if not label_file.exists():
                    # Tạo file label trống
                    label_file.touch()

if __name__ == "__main__":
    create_empty_labels()
    print("Đã tạo file label trống cho các ảnh chưa có annotation")
'''
    
    with open('create_labels.py', 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print("Đã tạo script create_labels.py để hỗ trợ tạo labels")

def main():
    print("=== Chuẩn bị Dataset Fire and Smoke từ Kaggle ===\n")
    
    # Bước 1: Kiểm tra và hướng dẫn cài đặt Kaggle API
    try:
        import kaggle
        print("✓ Kaggle API đã được cài đặt")
    except ImportError:
        setup_kaggle_api()
        return
    
    # Bước 2: Tải dataset
    if download_dataset():
        print("✓ Tải dataset thành công")
    else:
        print("✗ Lỗi khi tải dataset")
        return
    
    # Bước 3: Chuyển đổi sang định dạng YOLO
    try:
        convert_to_yolo_format()
        print("✓ Chuyển đổi dataset thành công")
    except Exception as e:
        print(f"✗ Lỗi khi chuyển đổi dataset: {str(e)}")
        return
    
    # Bước 4: Cập nhật file cấu hình
    update_data_yaml()
    print("✓ Cập nhật file cấu hình thành công")
    
    # Bước 5: Tạo script hỗ trợ
    create_better_labels()
    
    print("\n=== Hoàn thành chuẩn bị dataset ===")
    print("Bước tiếp theo:")
    print("1. Kiểm tra và cải thiện labels trong thư mục datasets/fire_smoke/labels/")
    print("2. Chạy huấn luyện: python train.py --data data.yaml --epochs 100")
    print("3. Sau khi huấn luyện xong, chuyển đổi sang Hailo: python convert_to_hailo.py --input runs/detect/fire_smoke_detector/weights/best.pt")

if __name__ == "__main__":
    main() 