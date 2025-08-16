import torch
import torch.nn as nn
import torchvision.models as models

class SwinModel(nn.Module):
    def __init__(self, num_classes):
        super(SwinModel, self).__init__()
        # Load Swin Transformer model from torchvision
        self.swin = models.swin_v2_t(weights="IMAGENET1K_V1")
        self.swin.head = nn.Linear(self.swin.head.in_features, num_classes)  # Modify classifier

    def forward(self, x):
        return self.swin(x)
