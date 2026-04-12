import os
import time
import numpy as np


class GestureRecorder:
    def __init__(self, save_dir="training/dataset", record_seconds=20):
        self.save_dir = save_dir
        self.record_seconds = record_seconds

        self.recording = False
        self.buffer = []
        self.start_time = None

        # 🔥 NEW: countdown state (non-blocking)
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

        label = input("Enter gesture name: ").strip()

        gesture_dir = os.path.join(self.save_dir, label)
        os.makedirs(gesture_dir, exist_ok=True)

        existing = len(os.listdir(gesture_dir))
        filename = f"sample_{existing + 1}.npy"

        path = os.path.join(gesture_dir, filename)

        data = np.array(self.buffer, dtype=np.float32)
        np.save(path, data)

        print(f"Saved: {path} | shape: {data.shape}")