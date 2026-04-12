import torch
import torch.nn as nn


class GestureModel(nn.Module):
    def __init__(self, input_size=159, hidden_size=128, num_classes=2):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)