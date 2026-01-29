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
from numpy.linalg import norm, solve
import pinocchio as pin
import numpy as np


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
        self.lb, self.ub = np.array(self.model.lowerPositionLimit), np.array(self.model.upperPositionLimit)
        self.joint_mid = (self.lb + self.ub) / 2.0

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

        for i in range(self.nq):
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
                # print("q = ", q)

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

        q = self._normalize_joint_angles(q) if success else None
        return success, q, final_err

    def dp_ik_constraint(self, target_pos, target_rot=None, q_init=None, eps=5e-4, max_iter=1000, dt=1e-1, damp=1e-12, joint_limit_margin=0.2, joint_limit_stiffness=200.0):
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
            q = np.zeros(self.nq)
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
            J = self.getJac(q)

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
            limit_penalty = np.zeros(self.nq)
            for i in range(self.nq):
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
            for i in range(self.nq):
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
    # 如果你有 urdf，使用: solver = PinocchioSolver("path/to/robot.urdf")
    solver = PinocchioSolver("./data/urdf/piper_no_gripper_description.urdf", verbose=False)
    joints = [0.03991598456487204, 2.3324827772046364, -1.5697804188440443, 0.32222799255209433, 0.9400658986956835, -0.21811814869317986]
    ee_pose = solver.fk(joints)
    print(ee_pose[:3, 3].round(5))
    ik_joints = solver.dp_ik_constraint(ee_pose[:3, 3], ee_pose[:3, :3], solver.joint_mid)
    print(ik_joints) if ik_joints is not None else print("Failed")
    fk_val = solver.fk(ik_joints) if ik_joints is not None else None
    delta_xyz = ee_pose[:3, 3] - fk_val[:3, 3] if fk_val is not None else None
    print(delta_xyz.round(5) * 1000) if delta_xyz is not None else print("No delta")
    jac = solver.getJac([0.0] * 6)
    print("jac =\n", jac.round(3))
