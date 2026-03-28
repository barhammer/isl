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

        # 🔥 THREAD POOL
        self.executor = ThreadPoolExecutor(max_workers=3)

        # 🔥 one processor per worker
        self.processors = [
            HandProcessor(MediaPipeHandler()),
            HandProcessor(MediaPipeHandler())
        ]

        # 🔥 one lock per processor
        self.locks = [threading.Lock() for _ in self.processors]

        self.scheduler = Scheduler()
        self.state = StateManager()

        # 🔥 skip limiter
        self.skip_counts = {}
        self.max_skip = 2

    # 🔥 worker function (UPDATED)
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

        # 🔥 SKIP FRAME → interpolate
        if not self.scheduler.should_run():
            interpolated = {}

            all_ids = set(self.state.curr_results) | set(self.state.prev_results)

            for obj_id in all_ids:
                curr = self.state.curr_results.get(obj_id, [])
                prev = self.state.prev_results.get(obj_id, [])

                hands = HandInterpolator.interpolate(prev, curr)
                interpolated[obj_id] = hands

                for hand_landmarks in hands:
                    self.mediapipe.mp_draw.draw_landmarks(
                        output,
                        hand_landmarks,
                        self.mediapipe.mp_hands.HAND_CONNECTIONS
                    )

            return output, interpolated

        # 🔥 GET CROPS (NEW STRUCTURE)
        crops, meta = self.cropper.get_crops(frame, id_to_box)

        if len(crops) == 0:
            return output, {}

        # 🔥 extract obj_ids
        obj_ids = [m[0] for m in meta]

        # 🔥 identify forced IDs
        forced_ids = []
        for obj_id in obj_ids:
            if self.skip_counts.get(obj_id, 0) >= self.max_skip:
                forced_ids.append(obj_id)

        # 🔥 LIMIT processing
        MAX_PEOPLE = 2
        active_ids = []

        # 1. forced first
        active_ids.extend(forced_ids)

        # 2. fill remaining
        remaining_slots = MAX_PEOPLE - len(active_ids)

        if remaining_slots > 0:
            remaining_ids = [i for i in obj_ids if i not in active_ids]

            if remaining_ids:
                for i in range(remaining_slots):
                    idx = (self.scheduler.person_index + i) % len(remaining_ids)
                    active_ids.append(remaining_ids[idx])

                self.scheduler.person_index += remaining_slots

        # 🔥 PREPARE TASKS (FIXED)
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

        # 🔥 PARALLEL EXECUTION
        results_iter = self.executor.map(self._process_one, tasks)

        new_results = {}

        for obj_id, hands in results_iter:
            if hands:
                new_results[obj_id] = hands

        # 🔥 UPDATE STATE
        results = self.state.update(new_results)

        # 🔥 UPDATE SKIP COUNTS
        processed_ids = set(new_results.keys())

        for obj_id in obj_ids:
            if obj_id in processed_ids:
                self.skip_counts[obj_id] = 0
            else:
                self.skip_counts[obj_id] = self.skip_counts.get(obj_id, 0) + 1

        # 🔥 CLEANUP
        for obj_id in list(self.skip_counts.keys()):
            if obj_id not in obj_ids:
                del self.skip_counts[obj_id]

        # 🔥 scheduler update
        self.scheduler.update_motion(
            self.state.prev_results,
            self.state.curr_results
        )

        # 🔹 DRAW
        for hands in results.values():
            for hand_landmarks in hands:
                self.mediapipe.mp_draw.draw_landmarks(
                    output,
                    hand_landmarks,
                    self.mediapipe.mp_hands.HAND_CONNECTIONS
                )

        return output, results