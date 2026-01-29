#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
"""
@File    :   ik_mix.py
@Time    :   2025/12/09 16:01:01
@Author  :   StarCheng
@Version :   1.0
@Site    :   https://star-cheng.github.io/Blog/
"""
from pinocchio.visualize import MeshcatVisualizer
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
from numpy.linalg import norm, solve
from tracikpy import TracIKSolver
import pinocchio as pin
import numpy as np
import time
import csv
import os


class SqhSolver(TracIKSolver):
    def __init__(self, urdf_file, base_link, tip_link, timeout=0.005, epsilon=1e-5, solve_type="Speed", Visualization=True):
        """
        IK求解器, 支持多次尝试和关节限位惩罚
        参数:
            urdf_file: URDF文件路径
            base_link: 基座链接名称
            tip_link: 末端链接名称
            timeout: IK求解超时时间
            epsilon: IK求解精度
            solve_type: IK求解类型, 可选值: Speed (default), Distance, Manipulation1, Manipulation2
        """
        super().__init__(urdf_file, base_link, tip_link, timeout, epsilon, solve_type)
        self.lb, self.ub = self.joint_limits
        self.joint_mid = (self.lb + self.ub) / 2

    def ik(self, ee_pose, qinit=None, bx=1e-5, by=1e-5, bz=1e-5, brx=1e-3, bry=1e-3, brz=1e-3):
        solution = super().ik(ee_pose, qinit, bx, by, bz, brx, bry, brz)
        return solution

    @staticmethod
    def _skew(vec):
        """
        将局部世界对齐坐标系下的雅可比矩阵转换为世界坐标系下的雅可比矩阵
        返回向量的反对称矩阵 [v]_x, 使得 [v]_x @ w = v * w
        """
        x, y, z = vec
        return np.array(
            [
                [0.0, -z, y],
                [z, 0.0, -x],
                [-y, x, 0.0],
            ]
        )

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

    def _normalize_joint_angles(self, q):
        """
        将关节角度归一化到限位内，考虑旋转关节的周期性（2π周期）
        
        对于旋转关节，如果角度超出限位，通过加减 2π 的整数倍来找到限位内的等价角度。
        例如：37.481 可以通过减去 12*π 得到 -0.218，这个值在限位内。
        
        Args:
            q: 关节角度数组
            
        Returns:
            q_normalized: 归一化后的关节角度数组
        """
        q_normalized = np.array(q, copy=True)
        two_pi = 2 * np.pi
        
        for i in range(self.number_of_joints):
            lb, ub = self.lb[i], self.ub[i]
            q_i = q[i]
            
            # 如果已经在限位内，直接跳过
            if lb <= q_i <= ub:
                continue
            
            # 计算限位范围
            limit_range = ub - lb
            
            # 如果限位范围 >= 2π，说明是连续旋转关节，直接模运算
            if limit_range >= two_pi - 1e-6:
                # 将角度归一化到 [lb, lb + 2π) 范围内
                q_normalized[i] = ((q_i - lb) % two_pi) + lb
                # 确保在限位内（处理边界情况）
                if q_normalized[i] > ub:
                    q_normalized[i] -= two_pi
            else:
                # 限位范围 < 2π，尝试通过加减 2π 的整数倍来找到限位内的等价角度
                if q_i < lb:
                    # 角度小于下界，尝试加上 2π 的整数倍
                    # 计算需要加上多少个 2π 才能达到或超过下界
                    n = int(np.ceil((lb - q_i) / two_pi))
                    # 尝试加上 n*2π
                    candidate = q_i + n * two_pi
                    if lb <= candidate <= ub:
                        q_normalized[i] = candidate
                    else:
                        # 如果加上 n*2π 后超出上界，尝试 (n-1)*2π
                        if n > 0:
                            candidate = q_i + (n - 1) * two_pi
                            if lb <= candidate <= ub:
                                q_normalized[i] = candidate
                            else:
                                # 无法找到限位内的等价角度，保持原值
                                q_normalized[i] = q_i
                        else:
                            q_normalized[i] = q_i
                else:  # q_i > ub
                    # 角度大于上界，尝试减去 2π 的整数倍
                    # 计算需要减去多少个 2π 才能达到或低于上界
                    n = int(np.ceil((q_i - ub) / two_pi))
                    # 尝试减去 n*2π
                    candidate = q_i - n * two_pi
                    if lb <= candidate <= ub:
                        q_normalized[i] = candidate
                    else:
                        # 如果减去 n*2π 后小于下界，尝试 (n-1)*2π
                        if n > 0:
                            candidate = q_i - (n - 1) * two_pi
                            if lb <= candidate <= ub:
                                q_normalized[i] = candidate
                            else:
                                # 无法找到限位内的等价角度，保持原值
                                q_normalized[i] = q_i
                        else:
                            q_normalized[i] = q_i
        
        return q_normalized

    def get_numerical_jacobian(self, q, eps=1e-6, frame="WORLD"):
        n = len(q)
        jacobian = np.zeros((6, n))

        # 计算当前位置和姿态
        T_current = self.fk(q[:self.number_of_joints])
        R_current = T_current[:3, :3]

        for i in range(n):
            # 中心差分
            q_plus = np.array(q, copy=True)
            q_minus = np.array(q, copy=True)
            q_plus[i] += eps
            q_minus[i] -= eps

            # 获取末端位姿
            T_plus = self.fk(q_plus)
            T_minus = self.fk(q_minus)

            # 位置部分 - 中心差分
            pos_plus = T_plus[:3, 3]
            pos_minus = T_minus[:3, 3]
            # J_v = delta_p / dq
            jacobian[:3, i] = (pos_plus - pos_minus) / (2 * eps)

            # 姿态部分 - 使用旋转矩阵的差分
            R_plus = T_plus[:3, :3]
            R_minus = T_minus[:3, :3]

            # 计算旋转矩阵的对数映射得到角速度
            # ΔR = R_current^T * R_plus ≈ exp([ω]×) ≈ I + [ω]×
            delta_R_plus = R_current[:3, :3].T @ R_plus
            delta_R_minus = R_current[:3, :3].T @ R_minus

            # 使用对数映射获取旋转向量
            # 注意：scipy的Rotation有更好的实现
            from scipy.spatial.transform import Rotation as R

            r_plus = R.from_matrix(delta_R_plus).as_rotvec()
            r_minus = R.from_matrix(delta_R_minus).as_rotvec()

            # 中心差分，并转换到世界坐标系
            omega_local = (r_plus - r_minus) / (2 * eps)
            omega_world = R_current[:3, :3] @ omega_local
            # J_ω = delta_ω / dq
            jacobian[3:, i] = omega_world

        if frame == "WORLD":
            X = np.eye(6)
            X[:3, 3:] = self._skew(T_current[:3, 3])
            jac_world = X @ jacobian
            return jac_world
        return jacobian

    def dp_ik(self, target_pos, target_rot=None, q_init=None, eps=5e-4, max_iter=1000, dt=1e-1, damp=1e-12):
        """
        使用雅可比转置法实现逆运动学

        Args:
            target_pos: 目标位置 (3,)
            target_rot: 目标旋转矩阵 (3,3), 可选
            q_init: 初始关节角度, 默认全0
            eps: 收敛阈值
            max_iter: 最大迭代次数
            dt: 步长
            damp: 阻尼系数

        Returns:
            q: 关节角度数组, 或None（如果失败）
        """
        if q_init is None:
            q = np.zeros(self.number_of_joints)
        else:
            q = np.array(q_init)

        for iteration in range(max_iter):
            # 当前末端位姿
            # link_poses = self.fk(q)
            T_current = self.fk(q)
            p_current = T_current[:3, 3]

            # 位置误差
            e_pos = target_pos - p_current

            # 姿态误差（如果指定了目标姿态）
            R_current = T_current[:3, :3]
            e_rot = self._rotation_error(target_rot, R_current)
            e = np.concatenate([e_pos, e_rot])

            # 检查收敛
            error_norm = np.linalg.norm(e)
            if error_norm < eps:
                print(f"逆运动学收敛于 {iteration} 次迭代, 误差: {error_norm:.6f}")
                # 返回前将角度归一化到限位内（考虑周期性）
                q_normalized = self._normalize_joint_angles(q)
                return q_normalized

            # 计算雅可比矩阵
            J = self.get_numerical_jacobian(q)

            # 雅可比转置法：dq = dt * J^T * e
            # 或阻尼最小二乘法：dq = dt * J^T * (J * J^T + damp^2 * I)^-1 * e
            if damp > 0:
                # 阻尼最小二乘法
                J_JT = J @ J.T
                I = np.eye(J_JT.shape[0])
                dq = dt * J.T @ np.linalg.solve(J_JT + damp**2 * I, e)
            else:
                # 雅可比转置法
                dq = dt * J.T @ e

            # 更新关节角度
            q = q + dq

            if iteration % 100 == 0:
                print(f"Iteration {iteration}: error = {error_norm:.6f}")

        print(f"逆运动学未收敛, 最终误差: {error_norm:.6f}")
        # 即使未收敛，也尝试归一化角度后返回（如果误差可接受）
        q_normalized = self._normalize_joint_angles(q)
        return q_normalized
    def dp_ik_constraint(self, target_pos, target_rot=None, q_init=None, eps=5e-4, max_iter=1000, dt=1e-1, damp=1e-12, 
              joint_limit_margin=0.2, joint_limit_stiffness=200.0):
        """
        使用雅可比转置法实现逆运动学，支持关节限位约束

        Args:
            target_pos: 目标位置 (3,)
            target_rot: 目标旋转矩阵 (3,3), 可选
            q_init: 初始关节角度, 默认全0
            eps: 收敛阈值
            max_iter: 最大迭代次数
            dt: 步长
            damp: 阻尼系数
            joint_limit_margin: 关节限位边界附近的阈值（弧度），用于软约束
            joint_limit_stiffness: 关节限位惩罚的刚度系数

        Returns:
            q: 关节角度数组, 或None（如果失败）
        """
        if q_init is None:
            q = np.zeros(self.number_of_joints)
        else:
            q = np.array(q_init)
        target_rot = np.array(target_rot) if target_rot is not None else None
        
        # 确保初始关节角度在限位内
        q = np.clip(q, self.lb, self.ub)
        for iteration in range(max_iter):
            # 当前末端位姿
            T_current = self.fk(q)
            p_current = T_current[:3, 3]

            # 位置误差
            e_pos = target_pos - p_current

            # 姿态误差（如果指定了目标姿态）
            R_current = T_current[:3, :3]
            if target_rot is not None:
                e_rot = self._rotation_error(target_rot, R_current)
            else:
                e_rot = np.zeros(3)
            e = np.concatenate([e_pos, e_rot])

            # 检查收敛
            error_norm = np.linalg.norm(e)
            if error_norm < eps:
                # 最终检查关节限位
                if np.all(q >= self.lb) and np.all(q <= self.ub):
                    print(f"逆运动学收敛于 {iteration} 次迭代, 误差: {error_norm:.6f}")
                    return q
                else:
                    # 如果超出限位，尝试在限位内寻找最近的有效解
                    q_clipped = np.clip(q, self.lb, self.ub)
                    T_clipped = self.fk(q_clipped)
                    p_clipped = T_clipped[:3, 3]
                    R_clipped = T_clipped[:3, :3]
                    e_pos_clipped = target_pos - p_clipped
                    if target_rot is not None:
                        e_rot_clipped = self._rotation_error(target_rot, R_clipped)
                    else:
                        e_rot_clipped = np.zeros(3)
                    error_clipped = np.linalg.norm(np.concatenate([e_pos_clipped, e_rot_clipped]))
                    if error_clipped < eps * 2:  # 允许稍大的误差
                        print(f"逆运动学收敛于 {iteration} 次迭代（限位内）, 误差: {error_clipped:.6f}")
                        return q_clipped

            # 计算雅可比矩阵
            J = self.get_numerical_jacobian(q)

            # 雅可比转置法：dq = dt * J^T * e
            # 或阻尼最小二乘法：dq = dt * J^T * (J * J^T + damp^2 * I)^-1 * e
            if damp > 0:
                # 阻尼最小二乘法
                J_JT = J @ J.T
                I = np.eye(J_JT.shape[0])
                dq = dt * J.T @ np.linalg.solve(J_JT + damp**2 * I, e)
            else:
                # 雅可比转置法
                dq = dt * J.T @ e

            # 关节限位约束处理
            # 1. 计算关节限位惩罚项（软约束）
            limit_penalty = np.zeros(self.number_of_joints)
            for i in range(self.number_of_joints):
                # 计算到限位边界的距离
                dist_to_lower = q[i] - self.lb[i]
                dist_to_upper = self.ub[i] - q[i]
                
                # 如果接近或超出下界
                if dist_to_lower < joint_limit_margin:
                    # 添加向限位中心回拉的力
                    limit_penalty[i] = joint_limit_stiffness * (self.joint_mid[i] - q[i]) * (1.0 - dist_to_lower / joint_limit_margin)
                # 如果接近或超出上界
                elif dist_to_upper < joint_limit_margin:
                    # 添加向限位中心回拉的力
                    limit_penalty[i] = joint_limit_stiffness * (self.joint_mid[i] - q[i]) * (1.0 - dist_to_upper / joint_limit_margin)
            
            # 将限位惩罚转换为关节空间的速度
            dq = dq + dt * limit_penalty * 0.1  # 限位惩罚的权重
            
            # 2. 限制步长，确保不会超出限位
            scale = 1.0
            for i in range(self.number_of_joints):
                if dq[i] > 0:
                    # 向上移动，检查上界
                    max_dq = self.ub[i] - q[i]
                    if max_dq < dq[i]:
                        scale = min(scale, max_dq / dq[i])
                elif dq[i] < 0:
                    # 向下移动，检查下界
                    max_dq = q[i] - self.lb[i]
                    if max_dq < -dq[i]:
                        scale = min(scale, max_dq / (-dq[i]))
            
            # 应用缩放因子，但保留至少很小的步长以保持收敛性
            scale = max(scale, 0.01)  # 最小步长比例
            dq = dq * scale

            # 更新关节角度
            q_new = q + dq
            
            # 3. 最终投影到限位内（作为安全措施）
            q = np.clip(q_new, self.lb, self.ub)

            if iteration % 100 == 0:
                print(f"Iteration {iteration}: error = {error_norm:.6f}")

        # 最终检查：如果解在限位内，即使误差稍大也返回
        q_final = np.clip(q, self.lb, self.ub)
        T_final = self.fk(q_final)
        p_final = T_final[:3, 3]
        R_final = T_final[:3, :3]
        e_pos_final = target_pos - p_final
        if target_rot is not None:
            e_rot_final = self._rotation_error(target_rot, R_final)
        else:
            e_rot_final = np.zeros(3)
        error_final = np.linalg.norm(np.concatenate([e_pos_final, e_rot_final]))
        
        if error_final < eps * 5:  # 允许稍大的误差
            print(f"逆运动学收敛（限位内）, 最终误差: {error_final:.6f}")
            return q_final
        
        print(f"逆运动学未收敛, 最终误差: {error_final:.6f}")
        return None


if __name__ == "__main__":
    # 1. 实例化 Solver
    solver = SqhSolver("./data/urdf/piper_no_gripper_description.urdf", "base_link", "link6", epsilon=1e-4, solve_type="Distance")
    joints = [0.03991598456487204, 2.3324827772046364, -1.5697804188440443, 0.32222799255209433, 0.9400658986956835, -0.21811814869317986]
    ee_pose = solver.fk(joints)
    print(ee_pose[:3, 3].round(5))
    ik_joints = solver.ik(ee_pose, [0.0] * 6)
    print(ik_joints)
    ik_joints = solver.dp_ik(ee_pose[:3, 3], ee_pose[:3, :3], solver.joint_mid)
    print(ik_joints.round(3))
    valid_ee = solver.fk(ik_joints)[:3, 3] if ik_joints is not None else None
    print("valid_ee = ", valid_ee.round(5)) if ik_joints is not None else print("valid_ee = None")
    print("delta = ", 1000 * (ee_pose[:3, 3] - valid_ee).round(5)) if ik_joints is not None else print("delta = None")
    print(solver.joint_limits)