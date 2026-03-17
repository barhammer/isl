class Scheduler:
    def __init__(self, skip=1, motion_threshold=0.005):
        self.base_skip = skip
        self.motion_threshold = motion_threshold

        self.frame_count = 0
        self.motion_score = 0.0

        self.person_index = 0
        self.people_per_frame = 1

    def should_run(self):
        self.frame_count += 1

        if self.motion_score > self.motion_threshold:
            effective_skip = 1
        else:
            effective_skip = self.base_skip

        return self.frame_count % effective_skip == 0

    def select_people(self, obj_ids):
        if len(obj_ids) <= 2:
            self.people_per_frame = 1
        else:
            self.people_per_frame = 2

        selected = []

        for i in range(self.people_per_frame):
            if obj_ids:
                idx = (self.person_index + i) % len(obj_ids)
                selected.append(obj_ids[idx])

        self.person_index += self.people_per_frame
        return selected

    def update_motion(self, prev_results, curr_results):
        motion = 0.0
        count = 0

        for obj_id in curr_results:
            curr = curr_results.get(obj_id, [])
            prev = prev_results.get(obj_id, [])

            for c_hand, p_hand in zip(curr, prev):
                for i in range(len(c_hand.landmark)):
                    dx = c_hand.landmark[i].x - p_hand.landmark[i].x
                    dy = c_hand.landmark[i].y - p_hand.landmark[i].y

                    motion += dx * dx + dy * dy
                    count += 1

        if count > 0:
            self.motion_score = motion / count