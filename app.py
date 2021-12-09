from threading import Thread
from comms import CommsThread, SharedData
from user import User
from camera import CamThread
import signal
import time


class App ():
    def __init__ (self):
        self.data = SharedData()
        signal.signal(signal.SIGINT, self.data.signal_handler)

        self.user = User()

        self.comms_thread = CommsThread(self.data)
        t1 = Thread(target=self.comms_thread.run)
        t1.start()

        self.cam_thread = CamThread(self.data)
        t2 = Thread(target=self.cam_thread.run)
        t2.start()

    def run(self):
        time.sleep(2)
        while True:
            img = None
            pose_dict = None
            with self.data.mx:
                img = self.data.image.copy()
                pose_dict = self.data.global_poses.copy()

            self.user.run(img, pose_dict, 
                self.comms_thread.send_end_pos_cmd,
                self.comms_thread.send_jaw_cmd)
            
            time.sleep(0.05)