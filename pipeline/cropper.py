import cv2
import numpy as np


class Cropper:
    def __init__(self, scale=1.8):
        self.scale = scale

    def get_crops(self, frame, id_to_box):
        """
        Args:
            frame: numpy array (H, W, 3)
            id_to_box: dict {obj_id: (x1, y1, x2, y2)}

        Returns:
            crops: list of images
            meta: list of (
                obj_id,
                (nx1, ny1, nx2, ny2),   # box in original frame
                (pad_x, pad_y),         # padding offset inside square
                (orig_w, orig_h)        # original (pre-pad) crop size
            )
        """
        crops = []
        meta = []

        h, w = frame.shape[:2]

        for obj_id, (x1, y1, x2, y2) in id_to_box.items():

            bw = x2 - x1
            bh = y2 - y1

            # 🔥 square crop size
            side = int(max(bw, bh) * self.scale)

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            nx1 = max(0, cx - side // 2)
            ny1 = max(0, cy - side // 2)
            nx2 = min(w, cx + side // 2)
            ny2 = min(h, cy + side // 2)

            crop = frame[ny1:ny2, nx1:nx2]

            if crop is None or crop.size == 0:
                continue

            orig_h, orig_w = crop.shape[:2]

            # 🔥 ensure square with CENTERED padding
            if orig_h != orig_w:
                size = max(orig_h, orig_w)
                square = np.zeros((size, size, 3), dtype=crop.dtype)

                pad_x = (size - orig_w) // 2
                pad_y = (size - orig_h) // 2

                square[pad_y:pad_y + orig_h, pad_x:pad_x + orig_w] = crop
                crop = square
            else:
                pad_x = 0
                pad_y = 0

            crops.append(crop)

            # 🔥 store everything needed for correct reverse mapping
            meta.append((
                obj_id,
                (nx1, ny1, nx2, ny2),
                (pad_x, pad_y),
                (orig_w, orig_h)
            ))

        return crops, meta