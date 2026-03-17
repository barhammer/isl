from ultralytics import YOLO

class Detector:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")  # lightweight

    def detect(self, frame):
        results = self.model(frame)
        return results