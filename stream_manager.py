import threading

import cv2


class StreamManager:
    """There is apparently no way to stop OpenCV from queueing up all incoming frames, so here we are grabbing and dropping them in a separate thread."""

    def __init__(self, src: str):
        self.cap = cv2.VideoCapture(src)

        if not self.cap.isOpened():
            raise IOError(f"Cannot open video stream at {src}")

        self.grabbed, self.frame = self.cap.read()
        self.running = self.grabbed
        self.lock = threading.Lock()

        if self.running:
            self.thread = threading.Thread(target=self.update, daemon=True)
            self.thread.start()

    def update(self):
        while self.running:
            grabbed, frame = self.cap.read()
            if grabbed:
                with self.lock:
                    self.frame = frame
            else:
                with self.lock:
                    self.running = False
                break

    def read(self):
        with self.lock:
            running = self.running
            frame = self.frame.copy()

        return running, frame

    def stop(self):
        with self.lock:
            self.running = False

        if hasattr(self, "thread"):
            self.thread.join(timeout=1)

        if self.cap.isOpened():
            self.cap.release()
