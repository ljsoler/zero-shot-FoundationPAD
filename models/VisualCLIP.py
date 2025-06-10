import torch
from torch import nn
import open_clip

class VisualCLIP(nn.Module):
    def __init__(self, model_name, num_classes = 1):
        super(VisualCLIP, self).__init__()
        self.backbone = open_clip.create_model_and_transforms(model_name.split('_')[1], pretrained="laion400m_e32")[0].visual
        # Freeze the backbone if you don't want to fine-tune it
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Add a new linear layer for classification
        self.classifier = nn.Linear(self.backbone.proj.shape[1], num_classes)

    def forward(self, x):
        # Extract features from the backbone
        with torch.no_grad():
            features = self.backbone(x)
        # Pass through the linear layer
        logits = self.classifier(features)
        return logits