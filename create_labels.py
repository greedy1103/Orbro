
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
