import os
import argparse
import time
import cv2
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(description='Fire and Smoke Detection using Hailo')
    parser.add_argument('--source', type=str, default=None, help='source (video file or webcam index or RTSP URL)')
    parser.add_argument('--model', type=str, required=True, help='path to Hailo HEF model file')
    parser.add_argument('--conf', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--save', action='store_true', help='save output video')
    parser.add_argument('--output', type=str, default='output_hailo.mp4', help='output video path')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--input-size', type=int, default=640, help='input size for the model')
    return parser.parse_args()

def preprocess_image(frame, input_size):
    """Preprocess image for Hailo model"""
    # Resize to model input size
    resized = cv2.resize(frame, (input_size, input_size))
    # Convert to RGB (if using OpenCV which is BGR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # Normalize to 0-1
    normalized = rgb.astype(np.float32) / 255.0
    # Transpose to channel-first format (NCHW)
    transposed = normalized.transpose(2, 0, 1)
    # Add batch dimension
    batched = np.expand_dims(transposed, axis=0)
    return batched

def postprocess_output(hailo_output, original_shape, input_size, conf_threshold=0.25):
    """Process Hailo model output to get bounding boxes, classes and scores"""
    # This function needs to be adapted based on the specific output format of your Hailo model
    # The following is a general template for YOLO models
    
    # Example assuming typical YOLO output
    # Extract outputs (this will depend on your specific model)
    height, width = original_shape[:2]
    
    # Process outputs (adjust according to your Hailo model's output format)
    boxes = []
    scores = []
    class_ids = []
    
    # Scale boxes to original image size
    scale_x = width / input_size
    scale_y = height / input_size
    
    # Assuming output has detections
    # This is placeholder code - you need to adapt based on your model's output format
    for detection in hailo_output[0]:  # Assuming first output contains detections
        # Example format: [x, y, w, h, confidence, class_id1, class_id2, ...]
        confidence = detection[4]
        
        if confidence >= conf_threshold:
            class_id = np.argmax(detection[5:])
            class_score = detection[5 + class_id]
            
            if class_score >= conf_threshold:
                cx, cy, w, h = detection[0:4]
                
                # Convert to top-left, bottom-right format
                x1 = (cx - w/2) * scale_x
                y1 = (cy - h/2) * scale_y
                x2 = (cx + w/2) * scale_x
                y2 = (cy + h/2) * scale_y
                
                boxes.append([x1, y1, x2, y2])
                scores.append(class_score)
                class_ids.append(class_id)
    
    return np.array(boxes), np.array(scores), np.array(class_ids)

def draw_detections(frame, boxes, scores, class_ids):
    """Draw detection boxes on frame"""
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box.astype(int)
        
        # Define color (red for fire, purple for smoke)
        color = (0, 0, 255) if class_id == 1 else (128, 0, 255)
        label = "Fire" if class_id == 1 else "Smoke"
        
        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        text_size = cv2.getTextSize(f"{label} {score:.2f}", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(frame, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
        
        # Draw label text
        cv2.putText(frame, f"{label} {score:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return frame

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Try to import Hailo Runtime
    try:
        from hailo_platform import HailoContext
    except ImportError:
        print("Error: Hailo Platform SDK not found. Please install it first.")
        return
    
    # Verify model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file {args.model} not found.")
        return
    
    print(f"Loading model: {args.model}")
    
    # Initialize Hailo device
    try:
        context = HailoContext()
        network = context.create_network_from_file(args.model)
        input_vstream = network.get_input_vstream()
        output_vstream = network.get_output_vstream()
        print("Hailo model loaded successfully")
    except Exception as e:
        print(f"Error initializing Hailo device: {str(e)}")
        return
    
    # Set up video source
    if args.source is None:
        print("No source specified. Using webcam 0")
        source = 0
    else:
        source = args.source
    
    print(f"Opening video source: {source}")
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
    
    # Statistics
    frame_count = 0
    fire_detected = 0
    smoke_detected = 0
    start_time = time.time()
    
    # Main loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video stream")
            break
        
        frame_count += 1
        loop_start = time.time()
        
        # Preprocess frame for Hailo
        input_data = preprocess_image(frame, args.input_size)
        
        # Run inference on Hailo
        input_vstream.send(input_data)
        hailo_output = output_vstream.receive()
        
        # Process detections
        boxes, scores, class_ids = postprocess_output(
            hailo_output, frame.shape, args.input_size, args.conf
        )
        
        # Update statistics
        if len(class_ids) > 0:
            fire_detected += (class_ids == 1).sum()
            smoke_detected += (class_ids == 0).sum()
        
        # Draw detections on frame
        result_frame = draw_detections(frame.copy(), boxes, scores, class_ids)
        
        # Calculate and display FPS
        process_time = time.time() - loop_start
        fps_text = f"FPS: {1.0/process_time:.1f}"
        cv2.putText(result_frame, fps_text, (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Add detection counts
        cv2.putText(result_frame, f'Fire: {fire_detected}', (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(result_frame, f'Smoke: {smoke_detected}', (20, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 0, 255), 2)
        
        # Display the result frame
        if args.view_img:
            cv2.imshow("Fire and Smoke Detection (Hailo)", result_frame)
        
        # Write to output video if enabled
        if out is not None:
            out.write(result_frame)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up resources
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    
    # Release Hailo resources
    context.close()
    
    # Print statistics
    total_time = time.time() - start_time
    print(f"\nDetection finished:")
    print(f"Processed {frame_count} frames in {total_time:.2f} seconds")
    print(f"Average FPS: {frame_count / total_time:.2f}")
    print(f"Total fire detections: {fire_detected}")
    print(f"Total smoke detections: {smoke_detected}")

if __name__ == '__main__':
    main() 