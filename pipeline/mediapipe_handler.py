import cv2
import mediapipe as mp


class MediaPipeHandler:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        # ✅ fast tracking mode (works with per-ID instances in runner)
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=20,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.hands.process(rgb)