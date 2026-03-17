import cv2

class Camera:
    def __init__(self, width=600, height=450, index=0):
        self.cap = cv2.VideoCapture(index)

        # 🔹 Try to set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.width = width
        self.height = height

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return False, None

        # 🔥 enforce resolution (fallback if camera ignores settings)
        frame = cv2.resize(frame, (self.width, self.height))

        return True, frame

    def release(self):
        self.cap.release()