import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models
from src.models.registry import register_model

@register_model("cnn_classifier")
class CNNClassifierPipeline:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = 12
        self.batch_size = 16
        self.epochs = 1
        
        # Initialize a lightweight CNN backbone for prototype
        print("Initializing final Stage 2 CNN Classifier...")
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
        self.model.to(self.device)
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        
    def _create_mock_dataloader(self):
        """Creates a dummy dataloader for prototype testing."""
        # Mocking 64 cropped patches (from Stage 1)
        mock_images = torch.randn(64, 3, 224, 224) 
        # Multi-label binary targets
        mock_labels = torch.randint(0, 2, (64, self.num_classes), dtype=torch.float32)
        dataset = TensorDataset(mock_images, mock_labels)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def run(self):
        print("Loading YOLO-cropped patches for CNN training (mocking data)...")
        dataloader = self._create_mock_dataloader()
        
        self.model.train()
        print(f"Starting CNN training for {self.epochs} epoch(s)...")
        
        for epoch in range(self.epochs):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(dataloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                print(f"  Batch {i+1}/{len(dataloader)} Loss: {loss.item():.4f}")
                
            print(f"Epoch {epoch+1} Complete. Avg Loss: {running_loss/len(dataloader):.4f}")
            
        print("Stage 2 CNN training prototype completed successfully!")
