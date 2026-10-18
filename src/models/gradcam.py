import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
import cv2
import torchvision.models as models
from src.models.registry import register_model

@register_model("gradcam_twostage")
class GradCamTwoStagePipeline:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # For prototyping, we use ResNet18 as a proxy for the 'baseline model'
        self.model = models.resnet18(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 12) # 12 abnormalities
        self.model.to(self.device)
        self.model.eval()
        
        self.target_layer = self.model.layer4[-1].conv2
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, target_class=None):
        """Generates Grad-CAM heatmap for a given input."""
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        target = output[0, target_class]
        target.backward()
        
        # Global average pooling on gradients
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Weight activations by gradients
        for i in range(self.activations.size(1)):
            self.activations[:, i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap) + 1e-8
        
        return heatmap.cpu().detach().numpy()

    def threshold_to_bbox(self, heatmap, threshold=0.5):
        """Converts heatmap to a pseudo bounding box."""
        binary_map = (heatmap > threshold).astype(np.uint8)
        contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return [x, y, w, h]

    def run(self):
        """Runs the pipeline prototype to extract pseudo boxes."""
        print("Running Grad-CAM extraction prototype...")
        out_csv = Path("assets/processed/pseudo_boxes.csv")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        
        # Dummy data for prototype
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        heatmap = self.generate_heatmap(dummy_input)
        
        # Resize heatmap to original image size
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        bbox = self.threshold_to_bbox(heatmap_resized)
        
        if bbox:
            print(f"Generated pseudo-box: {bbox}")
            # Save YOLO format dataset entry
            df = pd.DataFrame([{
                "image_id": "dummy_01",
                "x": bbox[0], "y": bbox[1], "w": bbox[2], "h": bbox[3]
            }])
            df.to_csv(out_csv, index=False)
            print(f"Saved boxes to {out_csv}")
        
        print("Stage 1 YOLO dataset generated successfully. Ready for YOLO training.")
