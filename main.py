from utils.camera import Camera
from pipeline.system import ISLPipeline
from pipeline.visualizer import Visualizer
import cv2


def main():
    camera = Camera()
    system = ISLPipeline()

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        frame_out, landmarks_per_id, id_to_box = system.process(frame)

        # 🔹 Debug
        print("Per-ID hands:", {k: len(v) for k, v in landmarks_per_id.items()})

        # 🔹 Draw
        Visualizer.draw_people(frame_out, id_to_box)
        Visualizer.draw_hand_ids(frame_out, landmarks_per_id)

        cv2.imshow("ISL", frame_out)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()