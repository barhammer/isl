import cv2

class Camera:
    def __init__(self, source=0, width=600, height=450):
        self.cap = cv2.VideoCapture(source)

        self.is_video = isinstance(source, str)

        # 🔹 Only force resolution for webcam
        if not self.is_video:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.width = width
        self.height = height

    def read(self):
        ret, frame = self.cap.read()

        # 🔁 Loop video if it ends
        if not ret and self.is_video:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()

        if not ret:
            return False, None

        # 🔹 Resize only if needed (mainly for webcam)
        h, w = frame.shape[:2]
        if (w, h) != (self.width, self.height):
            frame = cv2.resize(frame, (self.width, self.height))

        return True, frame

    def release(self):
        self.cap.release()