import torch
from training.dataset_loader import GestureDataset
from training.model import GestureModel

ds = GestureDataset()

x, y = ds[0]

print("Input sample shape:", x.shape)

x = torch.tensor(x).unsqueeze(0)  # (1, 30, 159)

model = GestureModel()

out = model(x)

print("Model output:", out)
print("Output shape:", out.shape)