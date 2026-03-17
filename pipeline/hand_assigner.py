import math


class HandState:
    def __init__(self, max_hands=2):
        self.max_hands = max_hands
        self.hands = []  # [(x, y)]

    def update(self, new_centers):
        # 🔥 simple smoothing (reduces jitter)
        if not self.hands:
            self.hands = new_centers[:self.max_hands]
            return

        smoothed = []

        for i, curr in enumerate(new_centers[:self.max_hands]):
            if i < len(self.hands):
                prev = self.hands[i]

                # 🔥 smoothing factor
                alpha = 0.7

                x = int(alpha * prev[0] + (1 - alpha) * curr[0])
                y = int(alpha * prev[1] + (1 - alpha) * curr[1])

                smoothed.append((x, y))
            else:
                smoothed.append(curr)

        self.hands = smoothed


class HandAssigner:
    def __init__(self):
        self.states = {}  # obj_id → HandState

    def get_hand_centers(self, hand_landmarks, frame_shape):
        centers = []

        if not hand_landmarks:
            return centers

        h, w = frame_shape[:2]

        for hand in hand_landmarks:
            xs = [lm.x for lm in hand.landmark]
            ys = [lm.y for lm in hand.landmark]

            cx = int(sum(xs) / len(xs) * w)
            cy = int(sum(ys) / len(ys) * h)

            centers.append((cx, cy))

        return centers

    def assign(self, hand_centers, id_to_box):
        # 🔹 init states
        for obj_id in id_to_box.keys():
            if obj_id not in self.states:
                self.states[obj_id] = HandState()

        # 🔹 remove dead IDs
        for obj_id in list(self.states.keys()):
            if obj_id not in id_to_box:
                del self.states[obj_id]

        temp_assignments = {obj_id: [] for obj_id in id_to_box.keys()}

        for (hx, hy) in hand_centers:
            assigned_id = None

            # 🔹 1. OVERLAP CHECK
            for obj_id, (x1, y1, x2, y2) in id_to_box.items():
                margin = 30

                if (x1 - margin) <= hx <= (x2 + margin) and \
                   (y1 - margin) <= hy <= (y2 + margin):
                    assigned_id = obj_id
                    break

            # 🔹 2. DISTANCE FALLBACK
            if assigned_id is None:
                min_dist = float("inf")

                for obj_id, (x1, y1, x2, y2) in id_to_box.items():
                    px = (x1 + x2) // 2
                    py = (y1 + y2) // 2

                    dist = (hx - px) ** 2 + (hy - py) ** 2

                    if dist < min_dist:
                        min_dist = dist
                        assigned_id = obj_id

            # 🔹 assign if valid
            if assigned_id is not None:
                temp_assignments[assigned_id].append((hx, hy))

        # 🔹 update states
        for obj_id, centers in temp_assignments.items():
            self.states[obj_id].update(centers)

        return {
            obj_id: state.hands
            for obj_id, state in self.states.items()
        }