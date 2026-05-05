import torch
import torch.nn as nn


class LSTMGestureModel(nn.Module):
    def __init__(self, input_size=162, hidden_size=64, num_classes=None):
        super().__init__()

        if num_classes is None:
            raise ValueError("num_classes must be provided")

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
            dropout=0.0
        )

        self.fc = nn.Sequential(
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)