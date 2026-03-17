class PriorityManager:
    def __init__(self):
        self.motion_scores = {}  # obj_id → score

    def update(self, prev_results, curr_results):
        scores = {}

        for obj_id in curr_results:
            curr = curr_results.get(obj_id, [])
            prev = prev_results.get(obj_id, [])

            motion = 0.0
            count = 0

            for c_hand, p_hand in zip(curr, prev):
                for i in range(len(c_hand.landmark)):
                    dx = c_hand.landmark[i].x - p_hand.landmark[i].x
                    dy = c_hand.landmark[i].y - p_hand.landmark[i].y

                    motion += dx * dx + dy * dy
                    count += 1

            scores[obj_id] = (motion / count) if count > 0 else 0.0

        self.motion_scores = scores

    def get_prioritized(self, obj_ids):
        # 🔥 sort by motion (descending)
        return sorted(
            obj_ids,
            key=lambda x: self.motion_scores.get(x, 0),
            reverse=True
        )