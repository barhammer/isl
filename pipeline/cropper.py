class Cropper:
    def __init__(self, scale=1.6):
        self.scale = scale

    def get_crops(self, frame, id_to_box):
        crops = {}
        h, w = frame.shape[:2]

        for obj_id, (x1, y1, x2, y2) in id_to_box.items():

            bw = x2 - x1
            bh = y2 - y1

            # 🔥 square size (important for MediaPipe)
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

            # 🔥 IMPORTANT: return both crop + coords
            crops[obj_id] = (crop, (nx1, ny1, nx2, ny2))

        return crops