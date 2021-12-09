import re
import struct
from enum import IntEnum
from typing import Union, Tuple, List, Optional
from cobs import cobs
from crcmod import crcmod
import logging
logger = logging.getLogger(__name__)


class BPLProtocol:
    """Class used to encode and decode BPL packets."""
    CRC8_FUNC = crcmod.mkCrcFun(0x14D, initCrc=0xFF, xorOut=0xFF)

    @staticmethod
    def packet_splitter(buff: bytes) -> Tuple[List[bytes], Optional[bytes]]:
        """
        Split packets coming in along bpl protocol, Packets are split at b'0x00'.
        :param buff: input buffer of bytes
        :return: List of bytes separated by 0x00, and a remaining bytes of an incomplete packet.
        """
        incomplete_packet = None
        packets = re.split(b'\x00', buff)
        if buff[-1] != b'0x00':
            incomplete_packet = packets.pop()
        return packets, incomplete_packet

    @staticmethod
    def parse_packet(packet_in: Union[bytes, bytearray]) -> Tuple[int, int, bytes]:
        """
        Parse the packet returning a tuple of [int, int, bytes].
        If unable to parse the packet, then return 0,0,b''.
        :param packet_in: bytes of a full packet
        :return: device_id, packet_id, data in bytes.
        """

        packet_in = bytearray(packet_in)

        if packet_in and len(packet_in) > 3:
            try:
                decoded_packet: bytes = cobs.decode(packet_in)
            except cobs.DecodeError as e:
                logger.warning(f"parse_packet(): Cobs Decoding Error, {e}")
                return 0, 0, b''

            if decoded_packet[-2] != len(decoded_packet):
                logger.warning(f"parse_packet(): Incorrect length: length is {len(decoded_packet)} "
                               f"in {[hex(x) for x in list(decoded_packet)]}")
                return 0, 0, b''
            else:
                if BPLProtocol.CRC8_FUNC(decoded_packet[:-1]) == decoded_packet[-1]:
                    rx_data = decoded_packet[:-4]

                    device_id = decoded_packet[-3]
                    packet_id = decoded_packet[-4]
                    rx_data = rx_data
                    return device_id, packet_id, rx_data
                else:
                    logger.warning(f"parse_packet(): CRC error in {[hex(x) for x in list(decoded_packet)]} ")
                    return 0, 0, b''
        return 0, 0, b''

    @staticmethod
    def encode_packet(device_id: int, packet_id: int, data: Union[bytes, bytearray]):
        """
         Encode the packet using the bpl protocol.
        :param device_id: Device ID
        :param packet_id: Packet ID
        :param data: Data in bytes
        :return: bytes of the encoded packet.
        """
        tx_packet = bytes(data)
        tx_packet += bytes([packet_id, device_id, len(tx_packet)+4])
        tx_packet += bytes([BPLProtocol.CRC8_FUNC(tx_packet)])
        packet: bytes = cobs.encode(tx_packet) + b'\x00'
        return packet

    @staticmethod
    def decode_floats(data: Union[bytes, bytearray]) -> List[float]:
        """
        Decode a received byte list, into a float list as specified by the bpl protocol
        Bytes are decoded into 32 bit floats.
        :param data: bytes, but be divisible by 4.
        :return: decoded list of floats
        """
        list_data = list(struct.unpack(str(int(len(data)/4)) + "f", data))
        return list_data

    @staticmethod
    def encode_floats(float_list: List[float]) -> bytes:
        """ Encode a list of floats into bytes
        Floats are encoded into 32 bits (4 bytes)
        :param float_list: list of floats
        :return: encoded bytes
        """
        data = struct.pack('%sf' % len(float_list), *float_list)
        return data


class PacketReader:
    """
    Packet Reader
    Helper class to read and decode incoming bytes and account for the incomplete packets.
    """
    incomplete_packets = b''

    def receive_bytes(self, data: bytes) -> List[Tuple[int, int, bytes]]:
        """
        Decodes packets.
        Accounts for reading incomplete bytes.
        :param data: input bytes
        :return: a list of of decoded packets (Device ID, Packet ID, data (in bytes))
        """
        # Receive data, and return a decoded packet
        packet_list = []
        encoded_packets, self.incomplete_packets = BPLProtocol.packet_splitter(self.incomplete_packets + data)
        if encoded_packets:
            for encoded_packet in encoded_packets:
                if not encoded_packet:
                    continue
                decoded_packet = BPLProtocol.parse_packet(encoded_packet)
                packet_list.append(decoded_packet)
        return packet_list


class UniqueValueEnum(IntEnum):
    def __init__(self, *args):
        cls = self.__class__
        if any(self.value == e.value for e in cls):
            a = self.name
            e = cls(self.value).name
            raise ValueError(
                "aliases not allowed in DuplicateFreeEnum:  %r --> %r"
                % (a, e))


class PacketID(UniqueValueEnum):
    MODE = 0x01
    VELOCITY = 0X02
    POSITION = 0x03
    OPENLOOP = 0x04
    CURRENT = 0x05
    ACCELERATION = 0x06
    VELOCITY_DEMAND_INNER = 0X07
    POSITION_DEMAND_INNER = 0X08
    CURRENT_DEMAND_DIRECT = 0x09
    TORQUE_OUTPUT = 0x0B
    POSITION_VELOCITY_DEMAND = 0x0C
    INDEXED_POSITION = 0x0D
    RELATIVE_POSITION = 0x0E
    AUTO_LIMIT_CURRENT_DEMAND = 0x0F
    ABS_POSITION_VELOCITY_DEMAND = 0x1E
    INDEXED_POSITION_OFFSET = 0x36
    DEMAND_TIMEOUT = 0x37
    AUTO_LIMIT_DEMAND = 0x38
    AUTO_LIMIT_AT_STARTUP = 0x3D
    POSITION_LIMIT = 0x10
    POSITION_LIMIT_FACTORY = 0x1A
    VELOCITY_LIMIT = 0x11
    VELOCITY_LIMIT_FACTORY = 0x1B
    CURRENT_LIMIT = 0x12
    CURRENT_LIMIT_FACTORY = 0x1C
    TORQUE_LIMIT = 0x1F
    POSITION_GAIN = 0x13
    VELOCITY_GAIN = 0x14
    CURRENT_GAIN = 0x15
    VELOCITY_LIMIT_INNER = 0x16
    VELOCITY_LIMIT_INNER_FACTORY = 0x1D
    POSITION_GAINS_INNER = 0x17
    VELOCITY_GAINS_INNER = 0x18
    CURRENT_GAINS_DIRECT = 0x19
    VELOCITY_CONSTRAINT = 0xB5
    POSITION_PARAMETERS = 0x20
    VELOCITY_PARAMETERS = 0x21
    VELOCITY_PARAMETERS_2 = 0xB7
    CURRENT_PARAMETERS = 0x22
    INPUT_VOLTAGE_PARAMETERS = 0x23
    VELOCITY_INNER_PARAMETERS = 0x24
    ACCELERATION_PARAMETERS = 0x3E
    ADAPTIVE_PARAMETERS = 0x9F
    OVERDRIVE_ENABLE = 0x25
    OVERDRIVE_PARAMETERS = 0x26
    TEMPERATURE_PARAMETERS = 0x27
    FACTORY_CLIMATE = 0x28
    CLIMATE_PARAMETERS = 0x29
    LINK_TRANSFORM = 0x2B
    LINK_ATTACHMENT_0 = 0x2C
    LINK_ATTACHMENT_1 = 0x2D
    LINK_ATTACHMENT_2 = 0x2E
    LINK_ATTACHMENT_3 = 0x2F
    LINK_END_EFFECTOR_OFFSET = 0x35
    MOTOR_PARAMETERS = 0x30
    MOTOR_OFFSET_ANGLE = 0x31
    OPENLOOP_PARAMETERS = 0x32
    MOTOR_PARAMETERS_2 = 0x39
    ETH_IP_ADDRESS = 0x33
    ETH_PORT = 0x34
    MAX_ACCELERATION = 0x40
    CURRENT_HOLD_THRESHOLD = 0x41
    COMPLIANCE_GAIN = 0x42
    COMPLIANCE_PARAMETERS = 0x44
    MODE_SETTINGS = 0x43
    CURRENT_HOLD_PARAMETERS = 0x45
    KM_CONFIGURATION = 0xA0
    KM_END_POS = 0xA1
    KM_END_VEL = 0xA2
    KM_BOX_OBSTACLE_00 = 0xA3
    KM_BOX_OBSTACLE_01 = 0xA4
    KM_BOX_OBSTACLE_02 = 0xA5
    KM_BOX_OBSTACLE_03 = 0xA6
    KM_BOX_OBSTACLE_04 = 0xA7
    KM_BOX_OBSTACLE_05 = 0xA8
    KM_CYLINDER_OBSTACLE_00 = 0xA9
    KM_CYLINDER_OBSTACLE_01 = 0xAA
    KM_CYLINDER_OBSTACLE_02 = 0xAB
    KM_CYLINDER_OBSTACLE_03 = 0xAC
    KM_CYLINDER_OBSTACLE_04 = 0xAD
    KM_CYLINDER_OBSTACLE_05 = 0xAE
    KM_COLLISION_FLAG = 0xAF
    KM_FLOAT_PARAMETERS = 0xB0
    KM_COLLISION_COORDS = 0xB1
    KM_JOINT_STATE = 0xB2
    KM_JOINT_STATE_REQUEST = 0xB3
    KM_DH_PARAMETERS_0 = 0xB8
    KM_DH_PARAMETERS_1 = 0xB9
    KM_DH_PARAMETERS_2 = 0xBA
    KM_DH_PARAMETERS_3 = 0xBB
    KM_DH_PARAMETERS_4 = 0xBC
    KM_DH_PARAMETERS_5 = 0xBD
    KM_DH_PARAMETERS_6 = 0xBE
    KM_DH_PARAMETERS_7 = 0xBF
    POSITION_GAINS_INTERNAL = 0xE3
    VELOCITY_LIMIT_INTERNAL = 0xE4
    KM_FLOAT_PARAMETERS_INTERNAL = 0xE5
    POSITION_LIMIT_INTERNAL = 0xE6
    DEVICE_TYPE_INTERNAL = 0xE7
    KM_DH_PARAMETERS_0_INTERNAL = 0xE8
    KM_DH_PARAMETERS_1_INTERNAL = 0xE9
    KM_DH_PARAMETERS_2_INTERNAL = 0xEA
    KM_DH_PARAMETERS_3_INTERNAL = 0xEB
    KM_DH_PARAMETERS_4_INTERNAL = 0xEC
    KM_DH_PARAMETERS_5_INTERNAL = 0xED
    KM_DH_PARAMETERS_6_INTERNAL = 0xEE
    KM_DH_PARAMETERS_7_INTERNAL = 0xEF
    KM_POS_LIMIT_TRANSLATE = 0xC0
    KM_VEL_LIMIT_TRANSLATE = 0xC1
    KM_POS_LIMIT_YAW = 0xC2
    KM_POS_LIMIT_PITCH = 0xC3
    KM_POS_LIMIT_ROLL = 0xC4
    KM_VEL_LIMIT_ROTATE = 0xC5
    KM_POS_GAINS_TRANSLATE = 0xC6
    KM_VEL_GAINS_TRANSLATE = 0xC7
    KM_POS_GAINS_ROTATE = 0xC8
    KM_VEL_GAINS_ROTATE = 0xC9
    VELOCITY_SETPOINT = 0xCA
    KM_JOINT_POS_0 = 0xD0
    KM_JOINT_POS_1 = 0xD1
    KM_JOINT_POS_2 = 0xD2
    KM_JOINT_POS_3 = 0xD3
    KM_JOINT_POS_4 = 0xD4
    KM_JOINT_POS_5 = 0xD5
    KM_JOINT_POS_6 = 0xD6
    KM_JOINT_POS_7 = 0xD7
    REQUEST = 0x60
    SERIAL_NUMBER = 0x61
    MODEL_NUMBER = 0x62
    VERSION = 0x63
    DEVICE_ID = 0x64
    INTERNAL_HUMIDITY = 0x65
    INTERNAL_TEMPERATURE = 0x66
    INTERNAL_PRESSURE = 0x6E
    DEVICE_TYPE = 0x67
    HARDWARE_STATUS = 0x68
    RUN_TIME = 0x69
    DEVICE_ID_FOR_SERIAL_NUMBER = 0x70
    DEVICE_ID_FOR_TIMESTAMP = 0x73
    ELECTRICAL_VERSION = 0x6A
    MECHANICAL_VERSION = 0x6B
    SOFTWARE_VERSION = 0x6C
    MASTER_ARM_PREFIX = 0x6D
    PACKET_PARAMETERS = 0xDE
    LED_OUTPUT = 0x75
    STATE_ESTIMATOR_STATUS = 0x71
    COMS_PROTOCOL = 0x80
    SUPPLY_VOLTAGE = 0x90
    POWER = 0x3A
    HEARTBEAT_SET = 0x91
    HEARTBEAT_FREQUENCY_SET = 0x92
    HEARTBEAT_INT_SET = 0x93
    HEARTBEAT_INT_FREQUENCY_SET = 0x94
    SAVE = 0x50
    LOAD = 0x51
    SET_DEFAULTS = 0x52
    FORMAT = 0x53
    CHANGE_PAGE = 0x54
    ICMU_WRITE_REGISTER = 0xE0
    ICMU_INNER_WRITE_REGISTER = 0x4F
    ICMU_PARAMETERS = 0xE1
    ICMU_INNER_PARAMETERS = 0x6F
    ICMU_RAW_STREAM = 0x85
    ICMU_READ_REGISTER = 0x83
    ICMU_INNER_READ_REGISTER = 0x84
    SET_PROTOCOL_BPSS = 0xF0
    SET_PROTOCOL_BPL = 0xF1
    SYSTEM_RESET = 0xFD
    BOOTLOADER_STM = 0xFE
    BOOTLOADER = 0xFF
    TEST_PACKET = 0xE2
    DEFAULT_OPERATING_MODE = 0x72
    IMU_READING = 0x3b
    IMU_GRAVITY = 0x3c
    LED_INDICATOR_DEMAND = 0x74
    BOOTLOADER_BATCH = 0xFC
    BOOT_BATCH_REPORT = 0xFB
    KM_END_VEL_LOCAL = 0xCB
    KM_END_POS_LOCAL = 0x96
    KM_END_VEL_CAMERA = 0x97
    KM_END_VEL_WORK = 0x98
    KM_MOUNT_POS_ROT = 0xB4
    KM_FLOAT_PARAMETERS_2 = 0x76
    EXPECTED_DEVICES = 0x2A
    POS_PRESET_GO = 0x55
    POS_PRESET_CAPTURE = 0x56
    POS_PRESET_SET_0 = 0x57
    POS_PRESET_SET_1 = 0x58
    POS_PRESET_SET_2 = 0x59
    POS_PRESET_SET_3 = 0x5A
    POS_PRESET_NAME_0 = 0x5B
    POS_PRESET_NAME_1 = 0x5C
    POS_PRESET_NAME_2 = 0x5D
    POS_PRESET_NAME_3 = 0x5E
    POS_PRESET_ENABLE_LOCAL = 0x5F
    POS_SEQ_SET_0 = 0xDA
    POS_SEQ_SET_1 = 0xDB
    POS_SEQ_SET_2 = 0xDC
    POS_SEQ_SET_3 = 0xDD
    POS_SEQ_PARAMETERS = 0xDF
    RC_BASE_VELOCITY_SCALE = 0x9A
    RC_BASE_POSITION_SCALE = 0x9B
    RC_BASE_OPTIONS = 0x9C
    RC_BASE_EXT_DEVICE_IDS = 0x9D
    RC_JOYSTICK_PARAMETERS = 0x9E
    JOYSTICK_VALUES = 0x79
    BUTTON_DEMAND = 0x7A
    MULTICAST = 0x3F
    ATI_FT_READING = 0xD8
    ATI_FT_MESSAGE = 0xD9
    PATH_PLANNING_PARAMETERS = 0x8F
    BUS_STATE = 0x95
    COMMS_INSTRUCTION = 0xCF
    ETH_PARAMETERS = 0x46
    ETH_SOCKET_0_PARAMETERS = 0x47
    ETH_SOCKET_1_PARAMETERS = 0x48
    ETH_SOCKET_2_PARAMETERS = 0x49
    ETH_SOCKET_3_PARAMETERS = 0x4A
    ETH_SOCKET_4_PARAMETERS = 0x4B
    ETH_SOCKET_5_PARAMETERS = 0x4C
    ETH_SOCKET_6_PARAMETERS = 0x4D
    ETH_SOCKET_7_PARAMETERS = 0x4E
    IP_DEVICE_ID_MAP = 0xB6
    SERIAL_PASSTHROUGH = 0xCC
    SERIAL_PARAMETERS = 0xCD
    POWER_SUPPLY_PARAMETERS = 0xCE
    CAN_BUS_STATE_SCHED = 0x77
    CONTROL_RESTRICTED = 0x78
    BACKUP_PARAMETERS = 0x7B
    DIAGNOSTICS = 0x7C
    WORK_FRAME = 0x7D
