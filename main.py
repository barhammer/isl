from utils.camera import Camera
from pipeline.system import ISLPipeline
from pipeline.visualizer import Visualizer
import cv2
import numpy as np
import torch
from training.model import GestureModel


def main():
    camera = Camera()
    system = ISLPipeline()

    # 🔥 Load model
    model = GestureModel()
    model.load_state_dict(torch.load("training/model.pth"))
    model.eval()

    # 🔥 Detection smoothing buffer
    detection_count = {}

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        frame_out, landmarks_per_id, id_to_box, vectors_per_id, sequences_ready = system.process(frame)

        for obj_id, seq in sequences_ready.items():

            # ----------------------------------------
            # 🔥 FIX 1: Ignore multi-hand inputs
            # ----------------------------------------
            num_hands = len(landmarks_per_id.get(obj_id, []))
            if num_hands != 1:
                detection_count[obj_id] = 0
                continue

            # ----------------------------------------
            # 🔹 Prepare input
            # ----------------------------------------
            seq_np = np.array(seq, dtype=np.float32)
            x = torch.tensor(seq_np).unsqueeze(0)

            # ----------------------------------------
            # 🔹 Model inference
            # ----------------------------------------
            with torch.no_grad():
                out = model(x)
                probs = torch.softmax(out, dim=1)

            pred = torch.argmax(probs).item()
            confidence = probs[0][pred].item()

            # ----------------------------------------
            # 🔥 FIX 2: Temporal smoothing
            # ----------------------------------------
            if pred == 1 and confidence > 0.8:
                detection_count[obj_id] = detection_count.get(obj_id, 0) + 1
            else:
                detection_count[obj_id] = 0

            # ----------------------------------------
            # 🔥 PRINT ONLY WHEN IT MATTERS
            # ----------------------------------------
            if detection_count[obj_id] == 1:
                print(f"Started detecting Peace (Conf: {confidence:.2f})")

            if detection_count[obj_id] == 3:
                print("✌️ PEACE DETECTED")

        # ----------------------------------------
        # 🔹 Draw
        # ----------------------------------------
        Visualizer.draw_people(frame_out, id_to_box)
        Visualizer.draw_hand_ids(frame_out, landmarks_per_id)

        cv2.imshow("ISL", frame_out)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()