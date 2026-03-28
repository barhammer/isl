from ultralytics import YOLO
import torch

class Detector:
    def __init__(self, imgsz=512):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.imgsz = imgsz

        self.model = YOLO("yolov8n.pt")
        self.model.to(self.device)

        # 🔥 Warmup (removes first-frame lag)
        if self.device == "cuda":
            dummy = torch.zeros((1, 3, imgsz, imgsz)).to(self.device)
            self.model.predict(dummy, device=0, half=True, verbose=False)

    def detect(self, frame):
        results = self.model(
            frame,
            device=0 if self.device == "cuda" else "cpu",
            half=(self.device == "cuda"),
            imgsz=self.imgsz,
            conf=0.4,          # adjust if needed
            classes=[0],       # 🔥 ONLY detect persons
            verbose=False
        )
        return results