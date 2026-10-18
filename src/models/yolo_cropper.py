import os
import yaml
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import numpy as np
import cv2
from src.models.registry import register_model

@register_model("yolo_cropper")
class YoloCropperPipeline:
    def __init__(self):
        self.csv_path = Path("assets/processed/pseudo_boxes.csv")
        self.dataset_dir = Path("assets/processed/yolo_dataset")
        self.image_size = 224
        
    def _prepare_dataset(self):
        """Converts pseudo_boxes.csv into YOLO directory format."""
        if not self.csv_path.exists():
            print(f"Warning: {self.csv_path} not found. Creating a dummy one for prototyping.")
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{"image_id": "dummy_01", "x": 98, "y": 63, "w": 64, "h": 161}]).to_csv(self.csv_path, index=False)

        df = pd.read_csv(self.csv_path)
        
        # Setup directories
        for split in ["train", "val"]:
            (self.dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
            
        # Write dataset yaml
        yaml_path = self.dataset_dir / "dataset.yaml"
        yaml_data = {
            "path": str(self.dataset_dir.absolute()),
            "train": "images/train",
            "val": "images/val",
            "names": {0: "knee"}
        }
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_data, f)
            
        # Convert bounding boxes to YOLO format (normalized cx, cy, w, h)
        # For prototype, we put everything in train and copy one to val
        for idx, row in df.iterrows():
            img_id = row['image_id']
            x, y, w, h = row['x'], row['y'], row['w'], row['h']
            
            # Normalize coordinates
            cx = (x + w / 2) / self.image_size
            cy = (y + h / 2) / self.image_size
            nw = w / self.image_size
            nh = h / self.image_size
            
            label_str = f"0 {cx} {cy} {nw} {nh}\n"
            
            # Create dummy images for prototyping if they don't exist
            for split in ["train", "val"]:
                img_path = self.dataset_dir / "images" / split / f"{img_id}.png"
                if not img_path.exists():
                    dummy_img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
                    cv2.imwrite(str(img_path), dummy_img)
                    
                label_path = self.dataset_dir / "labels" / split / f"{img_id}.txt"
                with open(label_path, "w") as f:
                    f.write(label_str)
                    
        return yaml_path

    def run(self):
        print("Preparing YOLO dataset...")
        yaml_path = self._prepare_dataset()
        print(f"Dataset configuration saved to {yaml_path}")
        
        print("Initializing YOLOv8n for Stage 1...")
        # Load a pre-trained ultra-lightweight YOLOv8 nano model
        model = YOLO('yolov8n.pt') 
        
        print("Starting training (prototype epoch=1)...")
        # Train for 1 epoch to verify the pipeline
        results = model.train(
            data=str(yaml_path),
            epochs=1,
            imgsz=self.image_size,
            device='cpu', # Force CPU for prototype speed if GPU is busy, or '0' for GPU
            project=str(self.dataset_dir / "runs"),
            name="knee_cropper"
        )
        print("Stage 1 YOLO cropper training prototype completed!")
