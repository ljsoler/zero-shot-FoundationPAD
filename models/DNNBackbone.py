import torch
from torchvision import models
from torch.nn import functional as F
from torch import nn
import pytorch_lightning as pl
from torcheval.metrics import BinaryAccuracy, MulticlassAccuracy
import open_clip
from .VisualCLIP import VisualCLIP

DINO_SIZE =  {
    "vitb14": (768, 384),
    "vits14": (384, 256),
    "vitl14": (1024, 384)
}

class DNNBackbone(pl.LightningModule):

    def __init__(self,
                 num_classes: int = 1,
                 model_name: str = 'resnet34',
                 hub: str = 'pytorch/vision',
                 head_only = True,
                 learning_rate = 0.0001,
                 **kwargs) -> None:
        super(DNNBackbone, self).__init__()
        
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        try:
            self.model = torch.hub.load(hub, model_name, weights="DEFAULT")
        except:
           self.model = VisualCLIP(model_name=model_name, num_classes=num_classes) if 'CLIP' in model_name else torch.hub.load(hub, model_name)

        self.set_parameter_requires_grad(self.model, feature_extracting=head_only)

        if('mobilenet_v3_large' in model_name or 'efficientnet' in model_name):
            self.model.classifier[-1] = nn.Linear(in_features=1280, out_features=num_classes)
        elif('resnet' in model_name):
            self.model.fc = nn.Linear(in_features=self.model.fc.in_features, out_features=num_classes)
        elif('densenet' in model_name):
            self.model.classifier = nn.Linear(in_features=1024, out_features=num_classes)
        elif('swin' in model_name):
            self.model.head = nn.Linear(in_features=self.model.head.in_features, out_features=num_classes)
        elif('dino' in model_name):
            in_features, out_features = DINO_SIZE[model_name.split('_')[1]] 
            self.model.head = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)
        elif('CLIP' in model_name):
            self.model.classifier = nn.Linear(in_features=self.model.backbone.proj.shape[1], out_features=num_classes, bias=True)

        self.model.num_classes = num_classes
        self.ce = nn.BCELoss() if num_classes == 1 else nn.CrossEntropyLoss()

        self.acc_metric = BinaryAccuracy(threshold=0.5) if num_classes == 1 else MulticlassAccuracy() 
        
    
    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False


    def forward(self, x):
        x = self.model(x)
        return x if self.num_classes > 1 else F.sigmoid(x)
    
    
    def training_step(self, batch, batch_idx):
        x, _, target = batch
        prob = self.forward(x)
        loss = self.ce(torch.squeeze(prob, dim=1), target)

        self.acc_metric.update(torch.squeeze(prob, dim=1), target)
        acc = self.acc_metric.compute()

        self.log_dict({"train_loss": loss, "train_acc": acc}, prog_bar=True)
        return loss
    

    def validation_step(self, batch, batch_idx):
        x, _, target = batch
        prob = self.forward(x)
        loss = self.ce(torch.squeeze(prob, dim=1), target)

        self.acc_metric.update(torch.squeeze(prob, dim=1), target)
        acc = self.acc_metric.compute()

        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, _, target = batch
        prob = self.forward(x)
        loss = self.ce(torch.squeeze(prob, dim=1), target)

        self.acc_metric.update(torch.squeeze(prob, dim=1), target)
        acc = self.acc_metric.compute()

        self.log_dict({"test_loss": loss, "test_acc": acc}, prog_bar=True)
        return loss
    

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)