import numpy as np


class GestureController:
    def __init__(self, history_size=10, high_thresh=0.02, low_thresh=0.008):
        """
        history_size: number of frames to track per person
        high_thresh: motion above → dynamic mode
        low_thresh: motion below → static mode
        """
        self.history_size = history_size
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh

        self.person_states = {}

    # ----------------------------------------
    # 🔹 Ensure state exists
    # ----------------------------------------
    def _init_person(self, obj_id):
        if obj_id not in self.person_states:
            self.person_states[obj_id] = {
                "mode": "static",
                "history": []
            }

    # ----------------------------------------
    # 🔹 Update motion history
    # ----------------------------------------
    def update(self, obj_id, vec):
        """
        vec: (162,) feature vector from pipeline
        """
        self._init_person(obj_id)

        state = self.person_states[obj_id]

        state["history"].append(vec)

        if len(state["history"]) > self.history_size:
            state["history"].pop(0)

        return self._compute_motion(state)

    # ----------------------------------------
    # 🔹 Compute motion score
    # ----------------------------------------
    def _compute_motion(self, state):
        history = state["history"]

        if len(history) < 2:
            return 0.0

        arr = np.array(history)

        diffs = np.abs(np.diff(arr, axis=0))
        motion = np.mean(diffs)

        return motion

    # ----------------------------------------
    # 🔹 Decide mode (with hysteresis)
    # ----------------------------------------
    def decide_mode(self, obj_id, motion):
        state = self.person_states[obj_id]

        if motion > self.high_thresh:
            state["mode"] = "dynamic"
        elif motion < self.low_thresh:
            state["mode"] = "static"
        # else → keep previous mode

        return state["mode"]

    # ----------------------------------------
    # 🔹 Select output
    # ----------------------------------------
    def select_output(self, obj_id, static_out, dynamic_out_available, dynamic_out):
        """
        static_out: model output (tensor)
        dynamic_out_available: bool
        dynamic_out: model output (tensor or None)
        """
        state = self.person_states[obj_id]

        if state["mode"] == "dynamic" and dynamic_out_available:
            return dynamic_out, "dynamic"
        else:
            return static_out, "static"

    # ----------------------------------------
    # 🔹 Cleanup (optional)
    # ----------------------------------------
    def cleanup(self, active_ids):
        """
        Remove stale IDs
        """
        to_remove = [pid for pid in self.person_states if pid not in active_ids]
        for pid in to_remove:
            del self.person_states[pid]