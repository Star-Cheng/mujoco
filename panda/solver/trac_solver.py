from tracikpy import TracIKSolver
import transforms3d as tf3d
import numpy as np


class PandaKinematics(TracIKSolver):
    def __init__(self, urdf_path, base_link, tip_link):
        super().__init__(urdf_path, base_link, tip_link)

    def create_pose(self, xyz_: list, quat_wxyz_=None):
        """创建一个Pose对象"""
        pose_ = np.eye(4)
        pose_[:3, 3] = xyz_
        if quat_wxyz_ is None:
            quat_wxyz_ = [1, 0, 0, 0]
            pose_[:3, :3] = tf3d.quaternions.quat2mat(quat_wxyz_)  # wxyz顺序
        else:
            pose_[:3, :3] = tf3d.quaternions.quat2mat(quat_wxyz_)  # wxyz顺序
        return pose_
    
if __name__ == "__main__":
    # 创建 Panda 机械臂的运动学求解器
    panda_kinematics = PandaKinematics("./description/panda_description/urdf/panda.urdf", "panda_link0", "panda_hand")
    # 设置目标位姿
    target_pos = [0.5, 0.0, 0.5]
    target_quat = [0.0, 0.0, 0.7071, 0.7071]  # 四元数表示
    init_joints = None
    ee_pose = panda_kinematics.create_pose(target_pos, target_quat)
    
    # 求解逆运动学
    joints = panda_kinematics.ik(ee_pose, init_joints)
    
    # 输出关节角度
    print("Calculated joint angles:", joints)
    
    # 正向运动学验证
    valid_pos = panda_kinematics.fk(joints)
    print("Forward kinematics result:", valid_pos)
