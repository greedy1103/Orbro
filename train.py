import os
import argparse
from ultralytics import YOLO

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train YOLOv8 model for Fire and Smoke Detection')
    parser.add_argument('--data', type=str, required=True, help='path to data.yaml file')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='model to start training from')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch', type=int, default=16, help='batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='image size')
    parser.add_argument('--name', type=str, default='fire_smoke_detector', help='name for the training run')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Verify data file exists
    if not os.path.exists(args.data):
        print(f"Error: Data file {args.data} not found.")
        return
    
    print(f"Starting training with the following parameters:")
    print(f"Data: {args.data}")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch}")
    print(f"Image size: {args.imgsz}")
    print(f"Run name: {args.name}")
    
    # Load model
    model = YOLO(args.model)
    
    # Start training
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        name=args.name,
        verbose=True
    )
    
    print("Training completed successfully!")
    print(f"Model saved at: runs/detect/{args.name}/weights/")

if __name__ == '__main__':
    main() 