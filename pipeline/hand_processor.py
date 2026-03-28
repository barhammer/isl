import cv2


class HandProcessor:
    def __init__(self, mediapipe_handler):
        self.mediapipe = mediapipe_handler

    def process_crop(self, crop, box, pad, crop_size, frame_shape):
        frame_h, frame_w = frame_shape
        x1, y1, x2, y2 = box
        pad_x, pad_y = pad
        orig_w, orig_h = crop_size

        if crop is None or crop.size == 0:
            return []

        crop_h, crop_w = crop.shape[:2]

        # resize for MediaPipe
        crop_resized = cv2.resize(crop, (224, 224))
        hand_results = self.mediapipe.process(crop_resized)

        if not hand_results or not hand_results.multi_hand_landmarks:
            return []

        hands = hand_results.multi_hand_landmarks

        # scale from 224 → padded crop
        scale_x = crop_w / 224
        scale_y = crop_h / 224

        for hand_landmarks in hands:
            for lm in hand_landmarks.landmark:
                # back to padded crop coordinates
                px = lm.x * 224 * scale_x
                py = lm.y * 224 * scale_y

                # 🔥 REMOVE padding offset
                px -= pad_x
                py -= pad_y

                # 🔥 clamp to original crop
                px = max(0, min(px, orig_w))
                py = max(0, min(py, orig_h))

                # map to full frame
                lm.x = (x1 + px) / frame_w
                lm.y = (y1 + py) / frame_h

        return hands