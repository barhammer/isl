import os
import numpy as np
import random


class GestureDataset:
    def __init__(
        self,
        data_dir="training/dataset",
        seq_len=30,
        stride=5,
        split="train",
        val_ratio=0.2,
        seed=42,
        label_map=None   # 🔥 IMPORTANT
    ):
        self.data = []
        self.labels = []
        self.seq_len = seq_len

        print(f"📂 Loading dataset from: {data_dir}")
        print(f"🔀 Mode: {split}")

        if not os.path.exists(data_dir):
            print("❌ Dataset directory does not exist!")
            return

        random.seed(seed)

        # ----------------------------------------
        # 🔥 LABEL MAP (shared system)
        # ----------------------------------------
        if label_map is None:
            labels = sorted([
                d for d in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, d))
            ])
            self.label_map = {label: i for i, label in enumerate(labels)}
            print("🏷️ Auto label_map:", self.label_map)
        else:
            self.label_map = label_map
            labels = sorted(self.label_map.keys())
            print("🏷️ Using provided label_map:", self.label_map)

        # ----------------------------------------
        # 🔥 PER-CLASS SPLIT (FIXED)
        # ----------------------------------------
        selected_files = []

        for label_name in labels:
            folder = os.path.join(data_dir, label_name)

            if not os.path.exists(folder):
                print(f"⚠️ Missing folder: {label_name}")
                continue

            label = self.label_map[label_name]

            files = [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.endswith(".npy")
            ]

            print(f"➡️ {label_name}: {len(files)} files")

            if len(files) == 0:
                continue

            random.shuffle(files)
            split_idx = int(len(files) * (1 - val_ratio))

            if split == "train":
                chosen = files[:split_idx]
            elif split == "val":
                chosen = files[split_idx:]
            else:
                raise ValueError("split must be 'train' or 'val'")

            print(f"   → using {len(chosen)} files for {split}")

            for f in chosen:
                selected_files.append((f, label))

        if len(selected_files) == 0:
            print("❌ No data selected")
            return

        print(f"📊 Total selected files: {len(selected_files)}")

        # ----------------------------------------
        # 🔥 SEQUENCE CREATION
        # ----------------------------------------
        for path, label in selected_files:
            seq = np.load(path)

            if len(seq) < seq_len:
                continue

            for i in range(0, len(seq) - seq_len + 1, stride):
                window = seq[i:i + seq_len]

                self.data.append(window)
                self.labels.append(label)

        self.data = np.array(self.data, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)

        # ----------------------------------------
        # 🔹 Stats
        # ----------------------------------------
        print("\n📊 LSTM DATASET STATS:")
        print("Total samples:", len(self.data))

        if len(self.data) > 0:
            print("Data shape:", self.data.shape)
            print("Labels shape:", self.labels.shape)
        else:
            print("❌ NO DATA LOADED")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]