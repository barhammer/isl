import numpy as np

class CentroidTracker:
    def __init__(self, max_distance=100, max_disappeared=10):
        self.next_id = 0
        self.objects = {}        # id → centroid
        self.disappeared = {}    # id → frames missing
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared

    def update(self, boxes):
        centroids = []

        for (x1, y1, x2, y2) in boxes:
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            centroids.append((cx, cy))

        # 🔹 No detections
        if len(centroids) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1

                if self.disappeared[obj_id] > self.max_disappeared:
                    del self.objects[obj_id]
                    del self.disappeared[obj_id]

            # 🔥 OPTIONAL: reset IDs when scene is empty
            if len(self.objects) == 0:
                self.next_id = 0

            return self.objects

        # 🔹 No existing objects
        if len(self.objects) == 0:
            for c in centroids:
                self.objects[self.next_id] = c
                self.disappeared[self.next_id] = 0
                self.next_id += 1
            return self.objects

        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        # 🔹 Distance matrix
        D = np.zeros((len(object_centroids), len(centroids)))

        for i, oc in enumerate(object_centroids):
            for j, c in enumerate(centroids):
                D[i, j] = np.linalg.norm(np.array(oc) - np.array(c))

        # 🔥 SORT matches by distance (stronger matching)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        new_objects = {}
        new_disappeared = {}

        # 🔹 Assign matches
        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            if D[row, col] > self.max_distance:
                continue

            obj_id = object_ids[row]
            new_objects[obj_id] = centroids[col]
            new_disappeared[obj_id] = 0

            used_rows.add(row)
            used_cols.add(col)

        # 🔹 Handle unmatched existing objects
        for row in range(len(object_centroids)):
            if row not in used_rows:
                obj_id = object_ids[row]
                self.disappeared[obj_id] += 1

                if self.disappeared[obj_id] <= self.max_disappeared:
                    new_objects[obj_id] = self.objects[obj_id]
                    new_disappeared[obj_id] = self.disappeared[obj_id]

        # 🔥 KEY FIX: prevent duplicate new IDs for close detections
        for col in range(len(centroids)):
            if col in used_cols:
                continue

            c = centroids[col]

            # check if this centroid is too close to an already assigned one
            duplicate = False
            for existing_c in new_objects.values():
                dist = np.linalg.norm(np.array(c) - np.array(existing_c))
                if dist < self.max_distance * 0.5:  # tighter threshold
                    duplicate = True
                    break

            if duplicate:
                continue

            new_objects[self.next_id] = c
            new_disappeared[self.next_id] = 0
            self.next_id += 1

        self.objects = new_objects
        self.disappeared = new_disappeared

        return self.objects