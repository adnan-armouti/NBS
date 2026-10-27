"""Compact HAR classifiers. 2D ResNet for mmWave (17,256,128); 1D ResNet for the
sequence modalities WiFi (270,1000) and RFID (23,148)."""
import torch
import torch.nn as nn


class Res2D(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.b = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(),
                               nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c))
        self.r = nn.ReLU()

    def forward(self, x):
        return self.r(x + self.b(x))


class HAR2D(nn.Module):
    """mmWave classifier: input (B, 17, 256, 128)."""
    def __init__(self, in_ch=17, n_classes=55, w=64):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_ch, w, 7, stride=2, padding=3),
                                  nn.BatchNorm2d(w), nn.ReLU(), nn.MaxPool2d(2))
        self.stages = nn.Sequential(
            Res2D(w), nn.Conv2d(w, 2 * w, 3, stride=2, padding=1), nn.BatchNorm2d(2 * w), nn.ReLU(),
            Res2D(2 * w), nn.Conv2d(2 * w, 4 * w, 3, stride=2, padding=1), nn.BatchNorm2d(4 * w), nn.ReLU(),
            Res2D(4 * w))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3),
                                  nn.Linear(4 * w, n_classes))

    def forward(self, x):
        return self.head(self.stages(self.stem(x)))


class Res1D(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.b = nn.Sequential(nn.Conv1d(c, c, 5, padding=2), nn.BatchNorm1d(c), nn.ReLU(),
                               nn.Conv1d(c, c, 5, padding=2), nn.BatchNorm1d(c))
        self.r = nn.ReLU()

    def forward(self, x):
        return self.r(x + self.b(x))


class HAR1D(nn.Module):
    """WiFi/RFID classifier: input (B, in_ch, T)."""
    def __init__(self, in_ch=270, n_classes=55, w=128):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(in_ch, w, 7, stride=2, padding=3),
                                  nn.BatchNorm1d(w), nn.ReLU(), nn.MaxPool1d(2))
        self.stages = nn.Sequential(
            Res1D(w), nn.Conv1d(w, 2 * w, 3, stride=2, padding=1), nn.BatchNorm1d(2 * w), nn.ReLU(),
            Res1D(2 * w), nn.Conv1d(2 * w, 2 * w, 3, stride=2, padding=1), nn.BatchNorm1d(2 * w), nn.ReLU(),
            Res1D(2 * w))
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.3),
                                  nn.Linear(2 * w, n_classes))

    def forward(self, x):
        return self.head(self.stages(self.stem(x)))


def build(modality, n_classes=55):
    if modality == "mmwave":
        return HAR2D(17, n_classes)
    if modality == "wifi":
        return HAR1D(270, n_classes)
    if modality == "rfid":
        return HAR1D(23, n_classes)
    raise ValueError(modality)
