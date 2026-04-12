from pipeline.mediapipe_handler import MediaPipeHandler
from pipeline.cropper import Cropper
from pipeline.hand_processor import HandProcessor
from pipeline.interpolator import HandInterpolator
from pipeline.scheduler import Scheduler
from pipeline.state_manager import StateManager

from concurrent.futures import ThreadPoolExecutor
import threading
import cv2


class MediaPipeRunner:
    def __init__(self):
        self.mediapipe = MediaPipeHandler()
        self.cropper = Cropper()

        self.executor = ThreadPoolExecutor(max_workers=3)

        self.processors = [
            HandProcessor(MediaPipeHandler()),
            HandProcessor(MediaPipeHandler())
        ]

        self.locks = [threading.Lock() for _ in self.processors]

        self.scheduler = Scheduler()
        self.state = StateManager()

        self.skip_counts = {}
        self.max_skip = 2

    def _process_one(self, args):
        obj_id, crop, box, pad, crop_size, frame_shape, worker_id = args

        processor = self.processors[worker_id]
        lock = self.locks[worker_id]

        with lock:
            hands = processor.process_crop(
                crop,
                box,
                pad,
                crop_size,
                frame_shape
            )

        return obj_id, hands

    def process(self, frame, id_to_box):
        output = frame.copy()
        frame_h, frame_w = frame.shape[:2]

        # ----------------------------------------
        # 🔥 SKIP FRAME → interpolate
        # ----------------------------------------
        if not self.scheduler.should_run():
            interpolated = {}
            frame_flags = {}

            all_ids = set(self.state.curr_results) | set(self.state.prev_results)

            for obj_id in all_ids:
                curr = self.state.curr_results.get(obj_id, [])
                prev = self.state.prev_results.get(obj_id, [])

                hands = HandInterpolator.interpolate(prev, curr)
                interpolated[obj_id] = hands

                # 🔥 NEW: mark interpolated
                frame_flags[obj_id] = 0.5

                for hand_landmarks in hands:
                    self.mediapipe.mp_draw.draw_landmarks(
                        output,
                        hand_landmarks,
                        self.mediapipe.mp_hands.HAND_CONNECTIONS
                    )

            return output, interpolated, frame_flags

        # ----------------------------------------
        # 🔥 GET CROPS
        # ----------------------------------------
        crops, meta = self.cropper.get_crops(frame, id_to_box)

        if len(crops) == 0:
            return output, {}, {}

        obj_ids = [m[0] for m in meta]

        # ----------------------------------------
        # 🔥 FORCED IDS
        # ----------------------------------------
        forced_ids = []
        for obj_id in obj_ids:
            if self.skip_counts.get(obj_id, 0) >= self.max_skip:
                forced_ids.append(obj_id)

        # ----------------------------------------
        # 🔥 HYBRID SCHEDULER
        # ----------------------------------------
        MAX_PEOPLE = 2
        active_ids = []

        active_ids.extend(forced_ids)

        def priority_score(obj_id, box):
            x1, y1, x2, y2 = box

            area = (x2 - x1) * (y2 - y1)
            cx = (x1 + x2) // 2
            center_dist = abs(cx - frame_w // 2)
            skip = self.skip_counts.get(obj_id, 0)

            return (
                0.0001 * area
                - 0.01 * center_dist
                + 2.0 * skip
            )

        scored = []
        for obj_id, box, _, _ in meta:
            scored.append((obj_id, priority_score(obj_id, box)))

        scored.sort(key=lambda x: x[1], reverse=True)
        priority_ids = [obj_id for obj_id, _ in scored]

        rr_ids = []
        if obj_ids:
            for i in range(len(obj_ids)):
                idx = (self.scheduler.person_index + i) % len(obj_ids)
                rr_ids.append(obj_ids[idx])

        self.scheduler.person_index += MAX_PEOPLE

        for pid in priority_ids:
            if pid not in active_ids:
                active_ids.append(pid)
            if len(active_ids) >= MAX_PEOPLE:
                break

        for rid in rr_ids:
            if rid not in active_ids:
                active_ids.append(rid)
            if len(active_ids) >= MAX_PEOPLE:
                break

        # ----------------------------------------
        # 🔥 PREPARE TASKS
        # ----------------------------------------
        tasks = []

        for i, (crop, (obj_id, box, pad, crop_size)) in enumerate(zip(crops, meta)):
            if obj_id not in active_ids:
                continue

            worker_id = i % len(self.processors)

            tasks.append((
                obj_id,
                crop,
                box,
                pad,
                crop_size,
                (frame_h, frame_w),
                worker_id
            ))

        # ----------------------------------------
        # 🔥 PARALLEL EXECUTION
        # ----------------------------------------
        results_iter = self.executor.map(self._process_one, tasks)

        new_results = {}

        for obj_id, hands in results_iter:
            if hands:
                new_results[obj_id] = hands

        # ----------------------------------------
        # 🔥 UPDATE STATE
        # ----------------------------------------
        results = self.state.update(new_results)

        # 🔥 NEW: mark real frames
        frame_flags = {}
        for obj_id in results:
            frame_flags[obj_id] = 1.0

        # ----------------------------------------
        # 🔥 UPDATE SKIP COUNTS
        # ----------------------------------------
        processed_ids = set(new_results.keys())

        for obj_id in obj_ids:
            if obj_id in processed_ids:
                self.skip_counts[obj_id] = 0
            else:
                self.skip_counts[obj_id] = self.skip_counts.get(obj_id, 0) + 1

        for obj_id in list(self.skip_counts.keys()):
            if obj_id not in obj_ids:
                del self.skip_counts[obj_id]

        # ----------------------------------------
        # 🔥 UPDATE SCHEDULER
        # ----------------------------------------
        self.scheduler.update_motion(
            self.state.prev_results,
            self.state.curr_results
        )

        # ----------------------------------------
        # 🔹 DRAW
        # ----------------------------------------
        for hands in results.values():
            for hand_landmarks in hands:
                self.mediapipe.mp_draw.draw_landmarks(
                    output,
                    hand_landmarks,
                    self.mediapipe.mp_hands.HAND_CONNECTIONS
                )

        return output, results, frame_flags