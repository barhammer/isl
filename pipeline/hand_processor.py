import cv2


class HandProcessor:
    def __init__(self, mediapipe_handler):
        self.mediapipe = mediapipe_handler

    def process_crop(self, crop, box, frame_shape):
        frame_h, frame_w = frame_shape
        x1, y1, x2, y2 = box

        if crop is None or crop.size == 0:
            return []

        # resize for performance
        crop_resized = cv2.resize(crop, (224, 224))

        hand_results = self.mediapipe.process(crop_resized)

        if not hand_results or not hand_results.multi_hand_landmarks:
            return []

        hands = hand_results.multi_hand_landmarks

        box_w = x2 - x1
        box_h = y2 - y1

        # map back to full frame
        for hand_landmarks in hands:
            for lm in hand_landmarks.landmark:
                lm.x = (x1 + lm.x * box_w) / frame_w
                lm.y = (y1 + lm.y * box_h) / frame_h

        return hands