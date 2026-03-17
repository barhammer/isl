class StateManager:
    def __init__(self, max_missing=3):
        self.prev_results = {}
        self.curr_results = {}
        self.missing_counts = {}
        self.max_missing = max_missing

    def update(self, new_results):
        results = {}

        # reset seen IDs
        for obj_id in new_results:
            self.missing_counts[obj_id] = 0
            results[obj_id] = new_results[obj_id]

        # handle missing IDs
        for obj_id in list(self.curr_results.keys()):
            if obj_id in new_results:
                continue

            self.missing_counts[obj_id] = self.missing_counts.get(obj_id, 0) + 1

            if self.missing_counts[obj_id] <= self.max_missing:
                results[obj_id] = self.curr_results[obj_id]
            else:
                self.missing_counts.pop(obj_id, None)

        self.prev_results = self.curr_results
        self.curr_results = results

        return results