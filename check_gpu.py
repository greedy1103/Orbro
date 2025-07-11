import torch

def check_gpu():
    """Kiểm tra xem PyTorch có thể sử dụng GPU hay không."""
    if torch.cuda.is_available():
        print("✅ GPU is available!")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print("\n=> Ultralytics/YOLOv8 sẽ tự động sử dụng GPU.")
    else:
        print("❌ GPU is not available. PyTorch is using the CPU.")
        print("\nLý do có thể là:")
        print("1. Bạn chưa cài đặt driver NVIDIA cho card đồ họa.")
        print("2. Bạn đã cài đặt phiên bản PyTorch chỉ hỗ trợ CPU.")
        print("3. CUDA Toolkit chưa được cài đặt hoặc không tương thích với PyTorch.")
        print("\nĐể khắc phục, hãy cài lại PyTorch với phiên bản hỗ trợ CUDA từ trang chủ PyTorch:")
        print("https://pytorch.org/get-started/locally/")

if __name__ == "__main__":
    check_gpu() 