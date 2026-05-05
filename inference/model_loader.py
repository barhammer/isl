import torch
from training.static_model import StaticGestureModel


# ----------------------------------------
# 🔹 Load static model + labels
# ----------------------------------------
def load_static_system(device):
    checkpoint = torch.load("training/static_model.pth", map_location=device)

    label_map_raw = checkpoint["label_map"]
    num_classes = len(label_map_raw)

    model = StaticGestureModel(
        input_size=162,
        num_classes=num_classes
    ).to(device)

    model.load_state_dict(checkpoint["model"])
    model.eval()

    # convert {"Hello": 0} → {0: "Hello"}
    label_map = {int(v): k for k, v in label_map_raw.items()}

    return model, label_map


# ----------------------------------------
# 🔹 Get NONE class index
# ----------------------------------------
def get_none_class(label_map):
    for k, v in label_map.items():
        if v.lower() == "none":
            return k
    return None