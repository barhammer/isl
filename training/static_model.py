import torch
import torch.nn as nn

class StaticGestureModel(nn.Module):
    def __init__(self, input_size=162, num_classes=None):
        super().__init__()

        if num_classes is None:
            raise ValueError("❌ num_classes must be provided")

        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)