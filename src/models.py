"""Modelos del Hito 1. Todos se separan en `encoder` (imagen -> vector) + `head` (vector -> logits),
para poder guardar y reutilizar el encoder en los hitos 2 y 3."""

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResidualBlock(nn.Module):
    """Dos convoluciones 3x3 con conexión residual y = F(x) + x (atajo 1x1 si cambia la forma)."""

    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.shortcut = nn.Identity()
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class CNNResidualEncoder(nn.Module):
    """CNN residual propia, entrenada desde cero. 128x128 -> vector de 256."""

    out_dim = 256

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
        self.layers = nn.Sequential(
            ResidualBlock(32, 32),
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 256, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return self.pool(self.layers(self.stem(x))).flatten(1)


class ResNet18Encoder(nn.Module):
    """ResNet18 de torchvision sin la capa fc. 128x128 -> vector de 512."""

    out_dim = 512

    def __init__(self, pretrained: bool):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.resnet18(weights=weights)
        net.fc = nn.Identity()
        self.net = net

    def forward(self, x):
        return self.net(x)


class GestureClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, n_classes: int, dropout: float = 0.2):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(encoder.out_dim, n_classes))

    def forward(self, x):
        return self.head(self.encoder(x))  # logits (CrossEntropyLoss aplica softmax internamente)


def build_model(cfg: dict, n_classes: int) -> GestureClassifier:
    arch = cfg["arch"]
    if arch == "cnn_residual":
        encoder = CNNResidualEncoder()
    elif arch == "resnet18":
        encoder = ResNet18Encoder(pretrained=cfg.get("pretrained", False))
    else:
        raise ValueError(arch)
    model = GestureClassifier(encoder, n_classes, cfg.get("dropout", 0.2))
    if cfg.get("freeze_encoder", False):
        for p in model.encoder.parameters():
            p.requires_grad = False
    return model
