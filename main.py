from pipeline.detector import Detector
from pipeline.tracker import CentroidTracker
from pipeline.detection_utils import extract_person_boxes
from pipeline.matcher import match_ids_to_boxes
from pipeline.mediapipe_runner import MediaPipeRunner

from utils.camera import Camera
import cv2


def main():
    camera = Camera()

    detector = Detector()
    tracker = CentroidTracker()

    mp_runner = MediaPipeRunner()

    # 🔥 NEW: YOLO scheduling state
    frame_count = 0
    yolo_skip = 2
    last_boxes = []
    last_id_to_box = {}

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        frame_count += 1

        # 🔥 adaptive YOLO skip
        num_people = len(last_id_to_box)

        if num_people <= 1:
            yolo_skip = 3
        elif num_people == 2:
            yolo_skip = 2
        else:
            yolo_skip = 1

        # 🔥 run YOLO only when needed
        if frame_count % yolo_skip == 0 or not last_boxes:
            boxes = extract_person_boxes(detector.detect(frame))
            objects = tracker.update(boxes)
            id_to_box = match_ids_to_boxes(objects, boxes)

            last_boxes = boxes
            last_id_to_box = id_to_box

        else:
            # 🔥 reuse previous detections
            objects = tracker.update(last_boxes)
            id_to_box = match_ids_to_boxes(objects, last_boxes)

        # 🔥 REAL pipeline
        frame_with_hands, landmarks_per_id = mp_runner.process(frame, id_to_box)

        # 🔹 Debug
        print("Per-ID hands:", {k: len(v) for k, v in landmarks_per_id.items()})

        # 🔹 Draw person boxes
        for obj_id, (x1, y1, x2, y2) in id_to_box.items():
            cv2.rectangle(frame_with_hands, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            cv2.putText(frame_with_hands, f"ID {obj_id}", (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 🔹 Draw hand → ID mapping
        for obj_id, hands in landmarks_per_id.items():
            for hand in hands:
                wrist = hand.landmark[0]

                hx = int(wrist.x * frame.shape[1])
                hy = int(wrist.y * frame.shape[0])

                cv2.putText(frame_with_hands, f"H:{obj_id}", (hx, hy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow("ISL", frame_with_hands)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()