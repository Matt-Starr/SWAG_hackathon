import cv2
import time
from comms import SharedData

CAM_URL = "rtsp://admin:@192.168.2.10:554/stream=0"


class CamThread:
    def __init__ (self, data: SharedData):
        self.data = data
        self.cam = None

    def run (self):
        self.cam = cv2.VideoCapture(CAM_URL)

        while not self.cam.isOpened():
            if self.data.stop_event.is_set():
                print("Shut down camera thread.")
                return

        print(f"Opened camera stream at {CAM_URL}")

        while True:
            if self.data.stop_event.is_set():
                print("Shut down camera thread.")
                return

            _, frame = self.cam.read()
            rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            with self.data.mx:
                self.data.image = rgbFrame
            
            time.sleep(0.01)