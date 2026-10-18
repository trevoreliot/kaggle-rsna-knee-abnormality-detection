import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models
from torch.utils.checkpoint import checkpoint
from src.models.registry import register_model

class RadImageNetResNet(nn.Module):
    """
    Wrapper for a ResNet backbone intended to load RadImageNet weights.
    For prototyping, we initialize with standard ResNet18 weights.
    """
    def __init__(self):
        super().__init__()
        # In a real run, we would load the RadImageNet state_dict here.
        # We use weights=None or standard weights for prototype.
        resnet = models.resnet18(weights=None)
        # Strip the classification head to get raw features (e.g., 512-dim for ResNet18)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
    def forward(self, x):
        # x is (B, C, H, W)
        x = self.feature_extractor(x) # (B, 512, 1, 1)
        return x.view(x.size(0), -1) # (B, 512)

class SequenceTransformer(nn.Module):
    def __init__(self, feature_dim=512, num_classes=12, num_layers=2, nhead=4):
        super().__init__()
        self.backbone = RadImageNetResNet()
        
        # Transformer for sequence aggregation
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=nhead,
            dim_feedforward=feature_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.fc = nn.Linear(feature_dim, num_classes)
        
    def extract_features(self, x):
        """Helper to run the backbone with gradient checkpointing."""
        # x shape: (B*S, C, H, W)
        # Using checkpointing to save VRAM
        return checkpoint(self.backbone, x, use_reentrant=False)

    def forward(self, x):
        # x shape: (B, S, C, H, W) where S is the number of slices in the sequence
        B, S, C, H, W = x.size()
        
        # Flatten batch and sequence dimensions to process through 2D backbone
        x_flat = x.view(B * S, C, H, W)
        
        # Extract features (use gradient checkpointing to fit in 16GB VRAM)
        features = self.extract_features(x_flat) # (B*S, feature_dim)
        
        # Reshape back to sequence format
        features = features.view(B, S, -1) # (B, S, feature_dim)
        
        # Pass through Transformer
        transformed = self.transformer(features) # (B, S, feature_dim)
        
        # Aggregate across the sequence dimension (Global Average Pooling)
        aggregated = transformed.mean(dim=1) # (B, feature_dim)
        
        # Final classification
        out = self.fc(aggregated) # (B, num_classes)
        return out


@register_model("2.5d_transformer")
class SequenceTransformerPipeline:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = 12
        self.batch_size = 4
        self.num_slices = 10  # Simulating 10 slices per study sequence
        self.epochs = 1
        
        print("Initializing 2.5D Sequence Transformer...")
        self.model = SequenceTransformer().to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        
        # Set up Mixed Precision scaler for BF16/FP16
        self.scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())
        
    def _create_mock_dataloader(self):
        """Creates a dummy dataloader for prototype testing."""
        # (Batch, Sequence, Channels, Height, Width)
        mock_sequences = torch.randn(16, self.num_slices, 3, 224, 224) 
        mock_labels = torch.randint(0, 2, (16, self.num_classes), dtype=torch.float32)
        dataset = TensorDataset(mock_sequences, mock_labels)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def run(self):
        print("Loading 2.5D sequence mock data...")
        dataloader = self._create_mock_dataloader()
        
        self.model.train()
        print(f"Starting Sequence Transformer training for {self.epochs} epoch(s) with Mixed Precision & Gradient Checkpointing...")
        
        for epoch in range(self.epochs):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(dataloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Mixed Precision Context (BF16/FP16)
                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                
                # Scale the loss and call backward
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                running_loss += loss.item()
                print(f"  Batch {i+1}/{len(dataloader)} Loss: {loss.item():.4f}")
                
            print(f"Epoch {epoch+1} Complete. Avg Loss: {running_loss/len(dataloader):.4f}")
            
        print("Pipeline B (2.5D Sequence Transformer) prototype completed successfully!")
