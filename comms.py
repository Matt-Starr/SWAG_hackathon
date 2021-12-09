
import socket
import time
from threading import Event, Lock
import numpy as np
import sys
from typing import List, Tuple, Dict
from bplprotocol import BPLProtocol, PacketID, PacketReader
from transforms import Transforms as T

TX2_IP_ADDR = "192.168.2.7"
TX2_PORT = 6789
TX2_DEVICE_ID = 0x0E
WRIST_DEVICE_ID = 0x02
JAW_DEVICE_ID = 0x01

REQUEST_FREQUENCY = 20


class SharedData ():
    def __init__ (self):
        self.mx = Lock()
        self.stop_event = Event()
        self.image: np.ndarray = np.zeros((720,1280,3), np.uint8)
        self.global_poses: Dict[str, np.ndarray] = {
            'camera_end_joint': [[0, 0, 0], [0, 0, 0, 1]],
            'end_effector_joint': [[0, 0, 0], [0, 0, 0, 1]],
        }

    def signal_handler(self, sig, frame):
        self.stop_event.set()
        sys.exit(0)


class CommsThread:
    def __init__ (self, data: SharedData):
        self.data = data
        self.addr = (TX2_IP_ADDR, TX2_PORT)
        self.pr = PacketReader()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(0)
        self.cmd_packets = []

    def run (self):
        req_end_pos_packet = BPLProtocol.encode_packet(TX2_DEVICE_ID, PacketID.REQUEST, bytes([PacketID.KM_END_POS]))
        req_wrist_pos_packet = BPLProtocol.encode_packet(WRIST_DEVICE_ID, PacketID.REQUEST, bytes([PacketID.POSITION]))

        next_req_time = time.time() + (1/REQUEST_FREQUENCY)

        while True:
            try:
                recv_bytes, _ = self.sock.recvfrom(4096)
            except BaseException:
                recv_bytes = b''
            
            if recv_bytes:
                packets = self.pr.receive_bytes(recv_bytes)
                self.parse_packets(packets)

            if time.time() >= next_req_time:
                next_req_time += (1/REQUEST_FREQUENCY)
                self.sock.sendto(req_end_pos_packet, self.addr)
                self.sock.sendto(req_wrist_pos_packet, self.addr)
                while len(self.cmd_packets) > 0:
                    packet = self.cmd_packets.pop(0)
                    self.sock.sendto(packet, self.addr)

            if self.data.stop_event.is_set():
                print("Shut down comms thread.")
                return

    def parse_packets (self, packets: List[Tuple[int, int, bytes]]):
        for device_id, packet_id, data in packets:
            if packet_id == PacketID.POSITION and device_id == WRIST_DEVICE_ID:
                wrist_pos = BPLProtocol.decode_floats(data)[0]
                with self.data.mx:
                    end_pose = self.data.global_poses['end_effector_joint']
                    cam_pose = T.get_camera_pose(end_pose, wrist_pos)
                    self.data.global_poses['camera_end_joint'] = cam_pose
            
            if packet_id == PacketID.KM_END_POS and device_id == TX2_DEVICE_ID:
                floats = BPLProtocol.decode_floats(data)
                print(f'\nFLOATS:\n{floats}\n')
                [t_end, r_end] = T.get_end_pose(floats)
                with self.data.mx:
                    self.data.global_poses['end_effector_joint'] = [t_end, r_end]

    def send_end_pos_cmd (self, pos, orient: np.ndarray = None):
        with self.data.mx:
            q = self.data.global_poses['end_effector_joint'][1] if orient is None else orient
            euler_xyz = T.quat_to_euler_xyz(q)
            ypr = [euler_xyz[2], euler_xyz[1], euler_xyz[0]]
            to_send = [pos[0], pos[1], pos[2], ypr[0], ypr[1], ypr[2]]
            
            data = BPLProtocol.encode_floats(to_send)
            print(f'\nDATA:\n{pos + ypr}\n')
            packet = BPLProtocol.encode_packet(TX2_DEVICE_ID, PacketID.KM_END_POS, data)
            self.cmd_packets.append(packet)

    def send_jaw_cmd (self, pos: float):
        with self.data.mx:
            data = BPLProtocol.encode_floats([pos])
            packet = BPLProtocol.encode_packet(JAW_DEVICE_ID, PacketID.POSITION, data)
            self.cmd_packets.append(packet)
