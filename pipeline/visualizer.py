import cv2


class Visualizer:
    @staticmethod
    def draw_people(frame, id_to_box):
        for obj_id, (x1, y1, x2, y2) in id_to_box.items():
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            cv2.putText(frame, f"ID {obj_id}", (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    @staticmethod
    def draw_hand_ids(frame, landmarks_per_id):
        for obj_id, hands in landmarks_per_id.items():
            for hand in hands:
                wrist = hand.landmark[0]

                h, w = frame.shape[:2]
                hx = int(wrist.x * w)
                hy = int(wrist.y * h)

                cv2.putText(frame, f"H:{obj_id}", (hx, hy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)