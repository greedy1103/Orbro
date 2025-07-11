import os
import argparse
import time
import cv2
import numpy as np
from ultralytics import YOLO

def parse_arguments():
    parser = argparse.ArgumentParser(description='Fire and Smoke Detection using YOLOv8')
    parser.add_argument('--source', type=str, default=None, help='source (video file or webcam index or RTSP URL)')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='model path')
    parser.add_argument('--conf', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--save', action='store_true', help='save output video')
    parser.add_argument('--output', type=str, default='output.mp4', help='output video path')
    parser.add_argument('--view-img', action='store_true', help='display results')
    return parser.parse_args()

def process_frame(frame, model, conf_threshold):
    # Perform detection
    results = model(frame, conf=conf_threshold)
    
    # Process results and draw bounding boxes
    annotated_frame = results[0].plot()
    
    # Display FPS and additional info
    fps_text = f'FPS: {1.0 / (time.time() - process_frame.prev_time):.2f}'
    process_frame.prev_time = time.time()
    
    cv2.putText(annotated_frame, fps_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return annotated_frame, results[0]

# Initialize the previous time for FPS calculation
process_frame.prev_time = time.time()

def main():
    args = parse_arguments()
    
    # Load model
    model = YOLO(args.model)
    print(f"Loaded model: {args.model}")
    
    # Set up video source
    if args.source is None:
        print("No source specified. Using webcam 0")
        source = 0
    else:
        source = args.source
        print(f"Using source: {source}")
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # Default to 30 fps if unable to determine
    
    # Initialize video writer if saving is enabled
    out = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"Saving output to {args.output}")
    
    print("Starting detection. Press 'q' to quit.")
    
    # Count for fire and smoke detections
    fire_count = 0
    smoke_count = 0
    frame_count = 0
    
    # Main detection loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video stream")
            break
        
        frame_count += 1
        
        # Process frame
        annotated_frame, result = process_frame(frame, model, args.conf)
        
        # Count fire and smoke instances
        if len(result.boxes) > 0:
            classes = result.boxes.cls.cpu().numpy().astype(int)
            for cls in classes:
                if cls == 0:  # Assuming class 0 is smoke
                    smoke_count += 1
                elif cls == 1:  # Assuming class 1 is fire
                    fire_count += 1
        
        # Add detection counts
        cv2.putText(annotated_frame, f'Fire: {fire_count}', (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(annotated_frame, f'Smoke: {smoke_count}', (20, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 0, 255), 2)
        
        # Display the resulting frame
        if args.view_img:
            cv2.imshow('Fire and Smoke Detection', annotated_frame)
        
        # Write to output video if saving is enabled
        if out is not None:
            out.write(annotated_frame)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    
    print(f"Detection finished. Processed {frame_count} frames")
    print(f"Total fire detections: {fire_count}")
    print(f"Total smoke detections: {smoke_count}")

if __name__ == '__main__':
    main() 