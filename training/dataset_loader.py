import os
import numpy as np


class GestureDataset:
    def __init__(self, data_dir="training/dataset", seq_len=30):
        self.data = []
        self.labels = []
        self.seq_len = seq_len

        label_map = {
            "None": 0,
            "Peace": 1
        }

        print(f"📂 Loading dataset from: {data_dir}")

        if not os.path.exists(data_dir):
            print("❌ Dataset directory does not exist!")
            return

        for label_name in os.listdir(data_dir):
            print(f"➡️ Found folder: {label_name}")

            if label_name not in label_map:
                print(f"⚠️ Skipping unknown label: {label_name}")
                continue

            label = label_map[label_name]
            folder = os.path.join(data_dir, label_name)

            files = os.listdir(folder)
            print(f"   📁 {label_name} contains {len(files)} files")

            for file in files:
                if not file.endswith(".npy"):
                    print(f"   ⚠️ Skipping non-npy file: {file}")
                    continue

                path = os.path.join(folder, file)
                seq = np.load(path)

                print(f"   ✅ Loaded {file} with shape {seq.shape}")

                if len(seq) < seq_len:
                    print(f"   ⚠️ Skipping {file} (too short)")
                    continue

                for i in range(len(seq) - seq_len):
                    window = seq[i:i + seq_len]
                    self.data.append(window)
                    self.labels.append(label)

        self.data = np.array(self.data, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)

        print("\n📊 FINAL DATASET STATS:")
        print("Total samples:", len(self.data))

        if len(self.data) > 0:
            print("Data shape:", self.data.shape)
            print("Labels shape:", self.labels.shape)
            print("Sample window shape:", self.data[0].shape)
        else:
            print("❌ NO DATA LOADED")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]