class Scheduler:
    def __init__(self, skip=2):
        self.base_skip = skip
        self.frame_count = 0

        self.person_index = 0
        self.people_per_frame = 1

    def should_run(self):
        self.frame_count += 1

        # 🔥 FIXED RATE → guarantees interpolation
        return self.frame_count % self.base_skip == 0

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
        # 🔥 keep this if you want later, but UNUSED for scheduling
        pass