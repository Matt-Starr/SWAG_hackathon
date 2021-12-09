from scipy.spatial.transform import Rotation as R
import numpy as np


class Transforms:
    def euler_xyz_to_quat (rot_xyz: np.ndarray) -> np.ndarray:
        r = R.from_euler('xyz', rot_xyz)
        return r.as_quat()

    def quat_to_euler_xyz (quat: np.ndarray) -> np.ndarray:
        r = R.from_quat(quat)
        return r.as_euler('xyz');

    def get_end_pose (km_end_pos: np.ndarray) -> np.ndarray:
        pos_xyz = np.array(km_end_pos[0:3])
        rot_xyz = np.array([km_end_pos[5], km_end_pos[4], km_end_pos[3]])

        r_end = R.from_euler('xyz', rot_xyz)
        t_end = pos_xyz + r_end.apply([90, 0, 0]) # put the end pos 90mm away from jaw base

        return [t_end, r_end.as_quat()]

    def get_camera_pose (end_pose: np.ndarray, wrist_pos: float) -> np.ndarray:
        t_end = end_pose[0]
        r_end = R.from_quat(end_pose[1])

        r_wrist = R.from_rotvec(np.array([wrist_pos, 0, 0]))
        r_tip = r_end*r_wrist

        t_tip_to_cam = 1000*np.array([-0.198, 0.020, 0.052])
        r_tip_to_cam = R.from_rotvec([0, 1.57, 0]) # change of axes
        t_tip_to_cam_rel = r_tip.apply(t_tip_to_cam)

        t_cam = t_end + t_tip_to_cam_rel
        r_cam = r_tip*r_tip_to_cam

        return [t_cam, r_cam.as_quat()]