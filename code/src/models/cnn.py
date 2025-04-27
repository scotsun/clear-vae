"""CNN Models."""

import torch
import torch.nn as nn


class SimpleCNNClassifier(nn.Module):
    def __init__(self, n_class: int = 10, in_channel: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channel, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(2048, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, n_class),
        )

    def forward(self, x):
        h = self.net(x)
        return self.cls_head(h)


class SimpleCNN64Classifier(SimpleCNNClassifier):
    def __init__(self, n_class: int = 4, in_channel: int = 3) -> None:
        super().__init__(n_class, in_channel)
        self.net = nn.Sequential(
            nn.Conv2d(in_channel, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Flatten(),
        )


class LAMCNNClassifier(SimpleCNNClassifier):
    def __init__(self, n_class: int = 10, in_channel: int = 1) -> None:
        super().__init__(n_class, in_channel)
        self.cls_head = nn.Linear(2048, n_class)


class LAMCNN64Classifier(SimpleCNN64Classifier):
    def __init__(self, n_class: int = 4, in_channel: int = 3) -> None:
        super().__init__(n_class, in_channel)
        self.cls_head = nn.Linear(2048, n_class)
