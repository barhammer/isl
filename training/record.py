import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
from utils.camera import Camera
from pipeline.system import ISLPipeline
from training.recorder import GestureRecorder


def main():
    camera = Camera()
    system = ISLPipeline()
    recorder = GestureRecorder(save_dir="training/dataset")

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        frame_out, landmarks_per_id, id_to_box, vectors_per_id, sequences_ready = system.process(frame)

        # 🔥 update recorder
        finished = recorder.update(vectors_per_id)
        if finished:
            recorder.save()

        cv2.imshow("Recording", frame_out)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            recorder.start_countdown()

        if key == 27:  # ESC
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()