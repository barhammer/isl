from pipeline.vectorizer import landmarks_to_vector


class ISLPipeline:
    def __init__(self):
        from pipeline.detector import Detector
        from pipeline.tracker import CentroidTracker
        from pipeline.detection_utils import extract_person_boxes
        from pipeline.matcher import match_ids_to_boxes
        from pipeline.mediapipe_runner import MediaPipeRunner

        self.detector = Detector()
        self.tracker = CentroidTracker()
        self.extract_person_boxes = extract_person_boxes
        self.match_ids_to_boxes = match_ids_to_boxes
        self.mp_runner = MediaPipeRunner()

        # YOLO scheduling
        self.frame_count = 0
        self.last_boxes = []
        self.last_id_to_box = {}

        # 🔥 NEW: sequence buffer
        self.sequence_buffer = {}   # obj_id → list of vectors
        self.max_seq_len = 30

    def process(self, frame):
        self.frame_count += 1

        num_people = len(self.last_id_to_box)

        if num_people <= 1:
            yolo_skip = 3
        elif num_people == 2:
            yolo_skip = 2
        else:
            yolo_skip = 1

        if self.frame_count % yolo_skip == 0 or not self.last_boxes:
            boxes = self.extract_person_boxes(self.detector.detect(frame))
            objects = self.tracker.update(boxes)

            id_to_box = self.match_ids_to_boxes(objects, boxes)

            self.last_boxes = boxes
            self.last_id_to_box = id_to_box

        else:
            objects = self.tracker.update(self.last_boxes)
            id_to_box = self.match_ids_to_boxes(objects, self.last_boxes)

        # ----------------------------------------
        # 🔹 MediaPipe
        # ----------------------------------------
        frame_out, landmarks_per_id, frame_flags = self.mp_runner.process(frame, id_to_box)

        # ----------------------------------------
        # 🔹 Vectorization
        # ----------------------------------------
        vectors_per_id = {}

        for obj_id, hands in landmarks_per_id.items():
            confidence = frame_flags.get(obj_id, 0.5)
            vec = landmarks_to_vector(hands, confidence)
            vectors_per_id[obj_id] = vec

        # ----------------------------------------
        # 🔹 Sequence Buffer
        # ----------------------------------------
        sequences_ready = {}

        for obj_id, vec in vectors_per_id.items():

            if obj_id not in self.sequence_buffer:
                self.sequence_buffer[obj_id] = []

            self.sequence_buffer[obj_id].append(vec)

            # keep fixed size
            if len(self.sequence_buffer[obj_id]) > self.max_seq_len:
                self.sequence_buffer[obj_id].pop(0)

            # check if ready
            if len(self.sequence_buffer[obj_id]) == self.max_seq_len:
                sequences_ready[obj_id] = self.sequence_buffer[obj_id].copy()

        # ----------------------------------------
        # 🔹 Debug (lightweight)
        # ----------------------------------------
        #for obj_id, vec in vectors_per_id.items():
        #    print(f"ID {obj_id} → len: {len(vec)}, conf: {vec[-1]}")

        #for obj_id, seq in sequences_ready.items():
         #   print(f"🔥 ID {obj_id} → sequence ready: ({len(seq)}, {len(seq[0])})")

        return frame_out, landmarks_per_id, id_to_box, vectors_per_id, sequences_ready