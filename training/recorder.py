import os
import time
import numpy as np


class GestureRecorder:
    def __init__(self, save_dir=None, record_seconds=4):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_dir = os.path.join(base_dir, "dataset")

        self.record_seconds = record_seconds
        self.current_label = None
        self.recording = False
        self.buffer = []
        self.start_time = None

        self.countdown_active = False
        self.countdown_start = None
        self.countdown_duration = 5

    # ----------------------------------------
    # 🔹 START COUNTDOWN (non-blocking)
    # ----------------------------------------
    def start_countdown(self):
        if self.recording or self.countdown_active:
            return  # prevent overlap

        print("\nGet Ready...")
        self.countdown_active = True
        self.countdown_start = time.time()

    # ----------------------------------------
    # 🔹 START RECORDING
    # ----------------------------------------
    def start_recording(self):
        self.buffer = []
        self.start_time = time.time()
        self.recording = True

    # ----------------------------------------
    # 🔹 UPDATE (called every frame)
    # ----------------------------------------
    def update(self, vectors_per_id):
        # ----------------------------------------
        # 🔥 HANDLE COUNTDOWN
        # ----------------------------------------
        if self.countdown_active:
            elapsed = time.time() - self.countdown_start
            remaining = int(self.countdown_duration - elapsed)

            if remaining > 0:
                print(f"Get Ready... {remaining}", end="\r")
                return False
            else:
                print("START            ")
                self.countdown_active = False
                self.start_recording()

        # ----------------------------------------
        # 🔥 RECORDING
        # ----------------------------------------
        if not self.recording:
            return False

        if len(vectors_per_id) == 0:
            return False

        # take first detected person
        obj_id = list(vectors_per_id.keys())[0]
        vec = vectors_per_id[obj_id]

        self.buffer.append(vec)

        # check if recording finished
        if time.time() - self.start_time >= self.record_seconds:
            self.recording = False
            print("\nRecording complete.")
            return True

        return False

    # ----------------------------------------
    # 🔹 SAVE DATA
    # ----------------------------------------
    def save(self):
        if len(self.buffer) == 0:
            print("No data recorded.")
            return
        if self.current_label is None:
            self.current_label = input("Enter gesture name: ").strip()

        label = self.current_label
        if not label:
            print("❌ Invalid label.")
            return

        gesture_dir = os.path.join(self.save_dir, label)
        os.makedirs(gesture_dir, exist_ok=True)

        # ----------------------------------------
        # 🔹 Find next available index safely
        # ----------------------------------------
        existing_files = [
            f for f in os.listdir(gesture_dir)
            if f.endswith(".npy") and f.startswith("sample_")
        ]

        indices = []
        for f in existing_files:
            try:
                idx = int(f.replace("sample_", "").replace(".npy", ""))
                indices.append(idx)
            except:
                continue

        next_index = max(indices) + 1 if indices else 1

        filename = f"sample_{next_index}.npy"
        path = os.path.join(gesture_dir, filename)

        # ----------------------------------------
        # 🔹 Save data
        # ----------------------------------------
        data = np.array(self.buffer, dtype=np.float32)
        np.save(path, data)

        print(f"✅ Saved: {path} | shape: {data.shape}")
        self.buffer = []