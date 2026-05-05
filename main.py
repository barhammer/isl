import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)
absl_logging.set_stderrthreshold("error")

import cv2
import torch

from utils.camera import Camera
from pipeline.system import ISLPipeline
from pipeline.visualizer import Visualizer

from inference.model_loader import load_static_system

from runtime.fps import FPSCounter
from collections import defaultdict


def main():
    # ----------------------------------------
    # 🔹 Device
    # ----------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # ----------------------------------------
    # 🔹 Load model
    # ----------------------------------------
    model, label_map = load_static_system(device)
    print("Loaded labels:", label_map)

    # ----------------------------------------
    # 🔹 Core components
    # ----------------------------------------
    camera = Camera(0)
    system = ISLPipeline()
    fps_counter = FPSCounter()

    delay = 33

    # 🔥 Flicker control
    STABILITY_THRESHOLD = 3

    last_preds = defaultdict(lambda: None)
    pred_counts = defaultdict(int)
    stable_labels = {}

    # ----------------------------------------
    # 🔁 Main loop
    # ----------------------------------------
    while True:
        ret, frame = camera.read()
        if not ret:
            break

        frame_out, landmarks_per_id, id_to_box, vectors_per_id, _ = system.process(frame)

        for obj_id, vec in vectors_per_id.items():

            # ----------------------------------------
            # 🔥 NO HAND RESTRICTION (FIXED)
            # ----------------------------------------
            if vec is None:
                continue

            # ----------------------------------------
            # 🔹 RAW TOP-1 PREDICTION
            # ----------------------------------------
            x = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                out = model(x)
                probs = torch.softmax(out, dim=1)[0]

            pred = torch.argmax(probs).item()
            confidence = probs[pred].item()
            label_name = label_map.get(pred, "UNKNOWN")

            # ----------------------------------------
            # 🔥 STABILITY LOGIC
            # ----------------------------------------
            if last_preds[obj_id] == pred:
                pred_counts[obj_id] += 1
            else:
                pred_counts[obj_id] = 1
                last_preds[obj_id] = pred

            if pred_counts[obj_id] >= STABILITY_THRESHOLD:
                stable_labels[obj_id] = (label_name, confidence)

            if obj_id not in stable_labels:
                stable_labels[obj_id] = (label_name, confidence)

            label_name, confidence = stable_labels[obj_id]

            # ----------------------------------------
            # 🔹 Draw result near person ID
            # ----------------------------------------
            if obj_id in id_to_box:
                x1, y1, x2, y2 = id_to_box[obj_id]

                text = f"ID {obj_id}: {label_name} ({confidence:.2f})"

                if confidence > 0.7:
                    color = (0, 255, 0)
                elif confidence > 0.4:
                    color = (0, 255, 255)
                else:
                    color = (0, 165, 255)

                cv2.putText(
                    frame_out,
                    text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

        # ----------------------------------------
        # 🔹 FPS
        # ----------------------------------------
        fps = fps_counter.update()

        # ----------------------------------------
        # 🔹 Draw visuals
        # ----------------------------------------
        Visualizer.draw_people(frame_out, id_to_box)
        Visualizer.draw_hand_ids(frame_out, landmarks_per_id)

        cv2.putText(
            frame_out,
            f"FPS: {fps}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

        cv2.imshow("ISL DEMO", frame_out)

        if cv2.waitKey(delay) & 0xFF == 27:
            break

    # ----------------------------------------
    # 🔹 Cleanup
    # ----------------------------------------
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()