import os
import argparse
import subprocess
import sys

def parse_arguments():
    parser = argparse.ArgumentParser(description='Convert YOLOv8 model to Hailo HEF format')
    parser.add_argument('--input', type=str, required=True, help='input model path (PyTorch .pt file)')
    parser.add_argument('--output', type=str, default=None, help='output HEF file path')
    parser.add_argument('--imgsz', type=int, default=640, help='image size')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Verify input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input model file {args.input} not found.")
        return
    
    # Setup output path if not provided
    if args.output is None:
        output_base = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f"{output_base}.hef"
    
    print(f"Converting model: {args.input} to {args.output}")
    print(f"Using image size: {args.imgsz}")
    
    # Step 1: Convert PyTorch model to ONNX format
    onnx_path = os.path.splitext(args.output)[0] + '.onnx'
    print(f"Step 1: Converting to ONNX format -> {onnx_path}")
    
    try:
        # Try to import the required modules
        from ultralytics import YOLO
        
        # Load the model
        model = YOLO(args.input)
        
        # Export to ONNX
        success = model.export(format="onnx", imgsz=args.imgsz)
        if not success:
            print("Error: Failed to export model to ONNX format.")
            return
        
        print(f"Successfully converted to ONNX format: {onnx_path}")
        
    except ImportError:
        print("Error: Required modules not found. Please install ultralytics.")
        return
    except Exception as e:
        print(f"Error during ONNX conversion: {str(e)}")
        return
    
    # Step 2: Convert ONNX to Hailo format using Hailo SDK (assumed to be installed)
    print(f"\nStep 2: Converting ONNX to Hailo HEF format -> {args.output}")
    print("\nThis step requires Hailo SDK to be installed and configured.")
    print("Command that would be executed (you need Hailo SDK installed):")
    print(f"hailo_model_compiler --input-model {onnx_path} --output-model {args.output}")
    
    # Check if hailo_model_compiler is available
    try:
        # This is a check command only - not executing the actual conversion
        subprocess.run(["which", "hailo_model_compiler"], 
                      stdout=subprocess.PIPE, 
                      stderr=subprocess.PIPE, 
                      check=True)
        
        print("\nHailo compiler found. You can run the conversion with:")
        print(f"hailo_model_compiler --input-model {onnx_path} --output-model {args.output}")
        
        # Uncomment the following lines to execute the actual conversion
        # print("\nExecuting conversion...")
        # subprocess.run(["hailo_model_compiler", 
        #                "--input-model", onnx_path,
        #                "--output-model", args.output], 
        #               check=True)
        # print(f"\nSuccessfully converted to Hailo HEF format: {args.output}")
        
    except subprocess.CalledProcessError:
        print("\nWarning: hailo_model_compiler not found in PATH.")
        print("Please install Hailo SDK and make sure it's in your PATH.")
        print("Then run the following command manually:")
        print(f"hailo_model_compiler --input-model {onnx_path} --output-model {args.output}")
    except Exception as e:
        print(f"\nError checking for Hailo compiler: {str(e)}")

if __name__ == '__main__':
    main() 