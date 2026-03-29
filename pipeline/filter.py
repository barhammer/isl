import math


class PersonFilter:
    def __init__(self,
                 min_width=120,
                 min_height=200,
                 aspect_ratio_range=(0.2, 1.0),
                 min_hits=2,
                 max_missed=5,
                 max_avg_speed=50):

        self.min_width = min_width
        self.min_height = min_height
        self.aspect_ratio_range = aspect_ratio_range

        self.min_hits = min_hits
        self.max_missed = max_missed
        self.max_avg_speed = max_avg_speed

        # state per ID
        self.tracks = {}  # id → state

    # ----------------------------------------
    # 📏 BASIC FILTERS
    # ----------------------------------------
    def _valid_size(self, box):
        x1, y1, x2, y2 = box
        return (x2 - x1) >= self.min_width and (y2 - y1) >= self.min_height

    def _valid_aspect_ratio(self, box):
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        if h == 0:
            return False
        ratio = w / h
        return self.aspect_ratio_range[0] < ratio < self.aspect_ratio_range[1]

    # ----------------------------------------
    # 🧠 SPEED
    # ----------------------------------------
    def _center(self, box):
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

    def _speed(self, b1, b2):
        c1 = self._center(b1)
        c2 = self._center(b2)
        return math.hypot(c2[0] - c1[0], c2[1] - c1[1])

    def _avg_speed(self, history):
        if len(history) < 2:
            return 0

        speeds = [
            self._speed(history[i - 1], history[i])
            for i in range(1, len(history))
        ]
        return sum(speeds) / len(speeds)

    # ----------------------------------------
    # 🚀 MAIN
    # ----------------------------------------
    def update(self, id_to_box, is_yolo_frame):
        output = {}

        # ----------------------------------------
        # UPDATE TRACKS
        # ----------------------------------------
        for obj_id, box in id_to_box.items():

            if obj_id not in self.tracks:
                self.tracks[obj_id] = {
                    "hits": 1,
                    "missed": 0,
                    "history": []
                }

            t = self.tracks[obj_id]

            # 🔥 ONLY COUNT HITS ON YOLO FRAMES
            if is_yolo_frame:
                t["hits"] += 1
                t["history"].append(box)

            t["missed"] = 0

            # ----------------------------------------
            # FILTERING
            # ----------------------------------------

            # size
            if not self._valid_size(box):
                continue

            # aspect ratio
            if not self._valid_aspect_ratio(box):
                continue

            # persistence (YOLO-based now)
            if t["hits"] < self.min_hits:
                output[obj_id] = box   # 👈 allow it through early
                continue

            # speed
            avg_speed = self._avg_speed(t["history"])
            if avg_speed > self.max_avg_speed:
                continue

            output[obj_id] = box

        # ----------------------------------------
        # HANDLE MISSING TRACKS
        # ----------------------------------------
        for obj_id in list(self.tracks.keys()):
            if obj_id not in id_to_box:
                self.tracks[obj_id]["missed"] += 1

                if self.tracks[obj_id]["missed"] > self.max_missed:
                    del self.tracks[obj_id]

        return output