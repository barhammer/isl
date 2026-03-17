import cv2
import numpy as np
import math

def show_grid(crops, size=200):
    resized = []
    print("GRID FUNCTION CALLED")

    for obj_id, crop in crops.items():
        if crop is None or crop.size == 0:
            continue

        img = cv2.resize(crop, (size, size)).copy()

        cv2.putText(img, f"ID {obj_id}", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        # 🔥 Add border (helps a LOT visually)
        img = cv2.copyMakeBorder(img, 2, 2, 2, 2,
                                 cv2.BORDER_CONSTANT, value=(0,255,0))

        resized.append(img)

    if not resized:
        blank = np.zeros((size, size, 3), dtype=np.uint8)
        cv2.putText(blank, "No detections", (10, size//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.imshow("Crops", blank)
        return

    cols = int(math.ceil(math.sqrt(len(resized))))
    rows = int(math.ceil(len(resized) / cols))

    grid = []
    idx = 0

    for _ in range(rows):
        row = []
        for _ in range(cols):
            if idx < len(resized):
                row.append(resized[idx])
            else:
                row.append(np.zeros((size+4, size+4, 3), dtype=np.uint8))
            idx += 1
        grid.append(np.hstack(row))

    grid_img = np.vstack(grid)

    cv2.imshow("Crops", grid_img)