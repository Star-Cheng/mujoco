#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
"""
@File    :   ik_dls.py
@Time    :   2025/12/09 16:20:17
@Author  :   StarCheng
@Version :   1.0
@Site    :   https://star-cheng.github.io/Blog/
"""
from pinocchio.visualize import MeshcatVisualizer
from numpy.linalg import norm, solve
from tracikpy import TracIKSolver
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import pinocchio as pin
import numpy as np
import time
import csv
import os


class PinocchioSolver:
    def __init__(self, urdf_path=None, package_dir=None, verbose=False):
        """
        初始化 Solver 类

        Args:
            urdf_path (str): URDF 文件路径。如果为 None, 则加载示例模型
            package_dir (str): URDF 资源包路径(如果有网格文件mesh需指定)
            verbose (bool): 是否打印调试信息。
        """
        self.verbose = verbose

        # 1. 加载模型
        if urdf_path:
            # 如果指定了 URDF，加载它
            if package_dir:
                self.model = pin.buildModelFromUrdf(urdf_path, package_dir)
            else:
                self.model = pin.buildModelFromUrdf(urdf_path)
        else:
            # 否则加载示例模型 (6自由度机械臂)
            print("[INFO] No URDF provided, using sample manipulator model.")
            self.model = pin.buildSampleModelManipulator()

        # 2. 创建 Data 对象 (用于存储计算过程中的中间变量)
        self.data = self.model.createData()

        # 3. 设置默认末端关节 ID (通常是最后一个关节)
        self.ee_joint_id = self.model.njoints - 1
        self.ee_joint_name = "joint6"

        # 4. 获取关节维度
        self.nq = self.model.nq  # 关节位置维度
        self.nv = self.model.nv  # 关节速度维度 (通常等于nq)

        if self.verbose:
            print(f"[INFO] Model loaded: {self.model.name}, End-Effector ID: {self.ee_joint_id}")

    def getJac(self, q):
        q = np.array(q)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        J = pin.computeFrameJacobian(self.model, self.data, q, self.model.getFrameId(self.ee_joint_name), pin.ReferenceFrame.WORLD)
        return J

    def fk(self, q):
        """
        正运动学求解 (Forward Kinematics)

        Args:
            q (np.ndarray): 关节角度数组

        Returns:
            tuple: (translation, rotation_matrix)
                   translation: np.array [x, y, z]
                   rotation_matrix: np.array 3x3
        """
        # 执行正运动学计算
        q = np.array(q)
        pin.forwardKinematics(self.model, self.data, q)

        # 更新关节在世界坐标系下的位置
        pin.updateFramePlacements(self.model, self.data)

        # 获取末端关节的位姿对象 (SE3)
        # oMi 表示 Object (Joint) in World (Origin)
        oMi = self.data.oMi[self.ee_joint_id]

        # return oMi.translation, oMi.rotation
        ee_pose = np.eye(4, dtype=np.float64)
        ee_pose[:3, :3] = oMi.rotation
        ee_pose[:3, 3] = oMi.translation
        return ee_pose

    def _rotation_error(self, R_desired, R_current):
        """计算旋转矩阵误差"""
        # R_error = R_desired * R_current^T
        R_error = R_desired @ R_current.T

        # 从旋转矩阵提取旋转向量
        angle = np.arccos(np.clip((np.trace(R_error) - 1) / 2, -1.0, 1.0))

        if np.isclose(angle, 0):
            return np.zeros(3)

        # 提取旋转轴
        axis = np.array([R_error[2, 1] - R_error[1, 2], R_error[0, 2] - R_error[2, 0], R_error[1, 0] - R_error[0, 1]]) / (2 * np.sin(angle))

        return axis * angle

    def ik(self, target_pos, target_rot=None, q_init=None, eps=5e-4, it_max=1000, dt=1e-1, damp=1e-12):
        """
        逆运动学求解 (Inverse Kinematics) - 使用阻尼最小二乘法

        Args:
            target_pos (np.ndarray): 目标位置 [x, y, z]
            target_rot (np.ndarray, optional): 目标旋转矩阵 3x3。如果为None, 则只解位置(3D)。
            q_init (np.ndarray, optional): 初始猜测关节角。默认为当前中性位或零位。
            eps (float): 收敛阈值。
            it_max (int): 最大迭代次数。
            dt (float): 积分步长。
            damp (float): 阻尼系数 (防止奇异性)。

        Returns:
            tuple: (success, q_solution, final_err)
        """
        # 1. 构造目标位姿 SE3 对象
        if target_rot is None:
            target_rot = np.eye(3)  # 默认为无旋转

        oMdes = pin.SE3(target_rot, np.array(target_pos))

        # 2. 初始化 q
        if q_init is None:
            q = pin.neutral(self.model)
        else:
            q = np.array(q_init)

        # 3. 迭代求解
        success = False
        final_err = 0.0

        for i in range(it_max):
            # --- A. 正运动学更新 ---
            pin.forwardKinematics(self.model, self.data, q)

            # --- B. 计算误差 ---
            # iMd: inverse of M (current) * d (desired) -> 差异变换矩阵
            iMd = self.data.oMi[self.ee_joint_id].actInv(oMdes)

            # 将 SE3 误差映射到 6D 向量 (李代数 se3)
            # vector: [v_x, v_y, v_z, w_x, w_y, w_z]
            err = pin.log(iMd).vector
            final_err = norm(err)

            # --- C. 检查收敛 ---
            if final_err < eps:
                success = True
                print("q = ", q)

                if self.verbose:
                    print(f"[IK] Converged at iter {i}, error: {final_err:.6f}")
                break

            # --- D. 计算雅可比矩阵 ---
            J = self.getJac(q)
            p_current = self.fk(q)[0:3, 3]
            R_current = self.fk(q)[0:3, :3]
            e_pos = target_pos - p_current
            e_rot = self._rotation_error(target_rot, R_current)
            e = np.concatenate([e_pos, e_rot])

            J_JT = J @ J.T
            I = np.eye(J_JT.shape[0])
            dq = dt * J.T @ np.linalg.solve(J_JT + damp**2 * I, e)
            q = q + dq

        if not success and self.verbose:
            print(f"[IK] Failed to converge after {it_max} iters. Final error: {final_err:.6f}")

        return success, q, final_err


if __name__ == "__main__":
    # 1. 实例化 Solver
    # 如果你有 urdf，使用: solver = PinocchioSolver("path/to/robot.urdf")
    solver = PinocchioSolver("./data/urdf/piper_no_gripper_description.urdf", verbose=False)
    joints = [0.03991598456487204, 2.3324827772046364, -1.5697804188440443, 0.32222799255209433, 0.9400658986956835, -0.21811814869317986]
    ee_pose = solver.fk(joints)
    print(ee_pose[:3, 3].round(5))
    succuss, ik_joints, err = solver.ik(ee_pose[:3, 3], ee_pose[:3, :3], [0.0] * 6)
    print(ik_joints) if succuss else print("Failed")
    fk_val = solver.fk(ik_joints)
    delta_xyz = ee_pose[:3, 3] - fk_val[:3, 3]
    print(delta_xyz.round(5)*1000)
    jac = solver.getJac([0.0] * 6)
    print("jac =\n", jac.round(3))
