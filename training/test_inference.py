import torch
import numpy as np
from training.model import GestureModel
from training.dataset_loader import GestureDataset


# load model
model = GestureModel()
model.load_state_dict(torch.load("training/model.pth"))
model.eval()

# load dataset
ds = GestureDataset()

# test on a few samples
# test random samples instead of first few
import random

indices = random.sample(range(len(ds)), 10)

for i in indices:
    x, y = ds[i]

    x = torch.tensor(x).unsqueeze(0)

    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)

    pred = torch.argmax(probs).item()
    confidence = probs[0][pred].item()

    print(f"True: {y}, Pred: {pred}, Conf: {confidence:.3f}")