from tracikpy import TracIKSolver
import pinocchio as pin
import numpy as np
from numpy.linalg import norm, solve
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

        # 4. 获取关节维度
        self.nq = self.model.nq  # 关节位置维度
        self.nv = self.model.nv  # 关节速度维度 (通常等于nq)

        if self.verbose:
            print(f"[INFO] Model loaded: {self.model.name}, End-Effector ID: {self.ee_joint_id}")

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
        pin.forwardKinematics(self.model, self.data, q)

        # 更新关节在世界坐标系下的位置
        pin.updateFramePlacements(self.model, self.data)

        # 获取末端关节的位姿对象 (SE3)
        # oMi 表示 Object (Joint) in World (Origin)
        oMi = self.data.oMi[self.ee_joint_id]

        return oMi.translation, oMi.rotation

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
                if self.verbose:
                    print(f"[IK] Converged at iter {i}, error: {final_err:.6f}")
                break

            # --- D. 计算雅可比矩阵 ---
            # 在局部坐标系(Local Frame)下计算雅可比
            J = pin.computeJointJacobian(self.model, self.data, q, self.ee_joint_id)

            # --- E. 雅可比修正 (Jlog6) ---
            # 因为误差是在李代数空间计算的，我们需要将几何雅可比转换以匹配 log 映射
            J = -np.dot(pin.Jlog6(iMd.inverse()), J)

            # --- F. 求解速度增量 (阻尼最小二乘法) ---
            # v = -J^T * (J * J^T + damp * I)^-1 * err
            # solve 函数解线性方程: Ax = b
            v = -J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))

            # --- G. 更新关节角度 ---
            # 在流形上积分: q_next = q + v * dt
            q = pin.integrate(self.model, q, v * dt)

        if not success and self.verbose:
            print(f"[IK] Failed to converge after {it_max} iters. Final error: {final_err:.6f}")

        return success, q, final_err


def read_csv_file(csv_file):
    """读取CSV文件并解析数据"""
    trajectory_data = []

    try:
        with open(csv_file, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                # 解析每一行的数据
                point = {}
                point["timestamp"] = float(row["timestamp"])

                # 提取每个关节的位置数据
                positions = []
                for joint_name in ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]:
                    pos_key = f"{joint_name}_pos"
                    if pos_key in row and row[pos_key]:
                        positions.append(float(row[pos_key]))
                    else:
                        positions.append(0.0)

                point["positions"] = positions
                trajectory_data.append(point)

    except FileNotFoundError:
        print(f"CSV file not found: {csv_file}")
    except Exception as e:
        print(f"Error reading CSV file: {e}")

    return trajectory_data


# --- 测试代码 ---
if __name__ == "__main__":
    # 1. 实例化 Solver
    # 如果你有 urdf，使用: solver = PinocchioSolver("path/to/robot.urdf")
    solver = PinocchioSolver("./data/urdf/piper_no_gripper_description.urdf", verbose=False)
    tracik = TracIKSolver("./data/urdf/piper_no_gripper_description.urdf", "base_link", "link6", epsilon=1e-4)
    trajectory_data = read_csv_file("/home/gym/code/ros2/fishbot/src/company/03agilex/solver/trajectory.csv")
    # 2. 定义一个测试目标
    print("q_target_truth = ", [0.0] * 6)
    for i in range(len(trajectory_data)):
        current_joints = trajectory_data[i]["positions"][:6]
        pinocc_pose = solver.fk(np.array(current_joints))
        tracik_pose = tracik.fk(np.array(current_joints))
        det_xyz = pinocc_pose[0] - tracik_pose[:3, 3]
        # print("pinocc_pose = ", pinocc_pose[0])
        # print("tracik_pose = ", tracik_pose[:3, 3])

        # 4. 测试 IK
        # 给一个完全不同的初始猜测 (例如全0)
        success, q_sol, err = solver.ik(target_pos=pinocc_pose[0], target_rot=pinocc_pose[1], q_init=current_joints, eps=1e-4, dt=0.005, damp=2e-2)
        tracik_joints = tracik.ik(tracik_pose)
        # print("Pinocchio IK: ", q_sol)
        # print("TracIK IK: ", tracik_joints)
        if success and tracik_joints is not None:
            pinocc_valid = solver.fk(q_sol)
            tracik_valid = tracik.fk(tracik_joints)
            det_xyz = pinocc_valid[0] - tracik_valid[:3, 3]
            det_rot = pinocc_valid[1] - tracik_valid[1:4, 3]
            if np.linalg.norm(det_xyz) > 0.0001:
                print("[INFO] IK det_xyz error: ", np.linalg.norm(det_xyz))
