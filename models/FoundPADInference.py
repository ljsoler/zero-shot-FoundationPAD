import torch

class FoundPADInference(torch.nn.Module):
    def __init__(self, backbone, num_classes=1):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes

    def forward(self, x):
        x = self.backbone(x)
        return torch.sigmoid(x) if self.num_classes == 1 else x
