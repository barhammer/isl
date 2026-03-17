import cv2
from utils.geometry import get_centroid

def draw_boxes_and_ids(frame, id_to_box):
    for obj_id, (x1, y1, x2, y2) in id_to_box.items():
        # draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # draw ID
        cx, cy = get_centroid((x1, y1, x2, y2))
        cv2.putText(frame, f"ID {obj_id}", (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)