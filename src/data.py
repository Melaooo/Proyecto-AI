from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["like", "dislike", "palm", "stop", "fist", "one", "peace", "ok", "call", "mute", "no_gesture"]
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet


def build_transforms(train: bool, img_size: int = 128):
    if train:
        return T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
            T.RandomHorizontalFlip(),  # mano izquierda/derecha: ninguna clase depende del lado
            T.RandomRotation(15),
            T.ColorJitter(0.3, 0.3, 0.3, 0.05),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])
    return T.Compose([T.Resize(img_size), T.ToTensor(), T.Normalize(MEAN, STD)])


class HandCropDataset(Dataset):
    """Recortes de manos generados por scripts/make_crops.py."""

    def __init__(self, split: str, transform=None):
        self.df = pd.read_csv(ROOT / f"data/splits/crops_{split}.csv")
        self.labels = torch.tensor([CLASSES.index(l) for l in self.df["label"]])
        self.paths = [ROOT / p for p in self.df["path"]]
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[i]
