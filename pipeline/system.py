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

        frame_out, landmarks_per_id = self.mp_runner.process(frame, id_to_box)

        return frame_out, landmarks_per_id, id_to_box