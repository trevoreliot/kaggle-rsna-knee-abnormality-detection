import argparse
from src.models.registry import get_model
import src.models.gradcam  # Ensure it registers
import src.models.yolo_cropper # Ensure it registers
import src.models.cnn_classifier # Ensure it registers
import src.models.sequence_transformer # Ensure it registers
import src.explain.raptor_gradcam  # Ensure it registers

def main():
    parser = argparse.ArgumentParser(description="RSNA Knee Abnormality Detection Pipeline")
    parser.add_argument("--pipeline", type=str, required=True, 
                        help="Pipeline to run, e.g., 'gradcam_twostage' or '2.5d_transformer'")
    args = parser.parse_args()
    
    print(f"Starting pipeline: {args.pipeline}")
    pipeline_class = get_model(args.pipeline)
    pipeline_instance = pipeline_class()
    pipeline_instance.run()

if __name__ == "__main__":
    main()
