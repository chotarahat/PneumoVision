import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def build_model(num_classes=2, pretrained=True, dropout=0.2):
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes)
    )
    return model
