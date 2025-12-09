#!/usr/bin/env python3
# https://github.com/unitreerobotics/xr_teleoperate/blob/main/teleop/robot_control/robot_arm_ik.py
# https://github.com/ccrpRepo/mocap_retarget/blob/master/src/mocap/src/robot_ik.py

import casadi
import meshcat.geometry as mg
import numpy as np
import pinocchio as pin
import time
from pinocchio import casadi as cpin
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.visualize import MeshcatVisualizer

# export PYTHONPATH=$CONDA_PREFIX/lib/python3.10/site-packages:$PYTHONPATH
import os
import sys


class WeightedMovingFilter:
    def __init__(self, weights, data_size=35):
        """
        加权移动平均滤波器

        Args:
            weights: 权重数组, 权重和应为1.0
            data_size: 数据维度
        """
        self._window_size = len(weights)
        self._weights = np.array(weights)
        assert np.isclose(np.sum(self._weights), 1.0), "权重和必须为1.0!"

        self._data_size = data_size
        self._filtered_data = np.zeros(self._data_size)
        self._data_queue = []

    def add_data(self, new_data):
        """
        添加新数据到滤波器

        Args:
            new_data: 新数据点
        """
        assert len(new_data) == self._data_size

        # 跳过重复数据
        if len(self._data_queue) > 0 and np.array_equal(new_data, self._data_queue[-1]):
            return

        # 维护滑动窗口
        if len(self._data_queue) >= self._window_size:
            self._data_queue.pop(0)

        self._data_queue.append(new_data)
        self._filtered_data = self._apply_filter()

    def _apply_filter(self):
        """应用滤波器"""
        if len(self._data_queue) == 0:
            return np.zeros(self._data_size)

        if len(self._data_queue) < self._window_size:
            # 窗口未满, 使用平均
            return np.mean(self._data_queue, axis=0)

        # 应用加权平均
        data_array = np.array(self._data_queue)
        filtered = np.zeros(self._data_size)

        for i in range(self._data_size):
            filtered[i] = np.sum(data_array[:, i] * self._weights)

        return filtered

    def reset(self):
        """重置滤波器"""
        self._data_queue = []
        self._filtered_data = np.zeros(self._data_size)

    @property
    def filtered_data(self):
        """获取滤波后的数据"""
        return self._filtered_data.copy()


class RobotArmIK:
    def __init__(self, urdf_path, Visualization=False, fps=120):
        """
        机械臂逆运动学求解器

        Args:
            urdf_path: URDF文件路径
            Visualization: 是否可视化
            fps: 可视化帧率
        """
        np.set_printoptions(precision=5, suppress=True, linewidth=200)
        self.Visualization = Visualization
        self.fps = fps

        # 加载机械臂模型
        self.robot = pin.RobotWrapper.BuildFromURDF(urdf_path)

        # 创建简化模型（不锁定任何关节）
        self.mixed_jointsToLockIDs = []
        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.mixed_jointsToLockIDs,
            reference_configuration=np.array([0.0] * self.robot.model.nq),
        )

        # 创建Casadi模型和数据进行符号计算
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()

        # 创建符号变量
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1)
        # 末端执行器目标位姿
        self.cTf_endeff = casadi.SX.sym("tf_endeff", 4, 4)

        # 前向运动学
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        # 获取末端执行器帧ID
        self.endeff_id = self.reduced_robot.model.getFrameId("joint6")  # 假设末端执行器帧名为"tool0"

        # 定义误差函数
        self.translational_error = casadi.Function(
            "translational_error",
            [self.cq, self.cTf_endeff],
            [self.cdata.oMf[self.endeff_id].translation - self.cTf_endeff[:3, 3]],
        )

        self.rotational_error = casadi.Function(
            "rotational_error",
            [self.cq, self.cTf_endeff],
            [cpin.log3(self.cTf_endeff[:3, :3] @ self.cdata.oMf[self.endeff_id].rotation.T)],
        )

        # 定义优化问题
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        self.var_q_last = self.opti.parameter(self.reduced_robot.model.nq)  # 用于平滑
        self.param_tf_endeff = self.opti.parameter(4, 4)

        # 计算各种成本
        self.translational_cost = casadi.sumsqr(self.translational_error(self.var_q, self.param_tf_endeff))
        self.rotation_cost = casadi.sumsqr(self.rotational_error(self.var_q, self.param_tf_endeff))
        self.regularization_cost = casadi.sumsqr(self.var_q)
        self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)

        # 设置优化约束和目标
        self.opti.subject_to(self.opti.bounded(self.reduced_robot.model.lowerPositionLimit, self.var_q, self.reduced_robot.model.upperPositionLimit))

        # 目标函数权重
        w_trans = 50.0  # 平移误差权重, 位置要准
        w_rot = 50.0  # 旋转误差权重, 角度要准
        w_reg = 0.001  # 正则化权重, 关节要尽量靠近零位
        w_smooth = 0.001  # 平滑权重, 动作要慢/平滑 (q - q_last)

        self.opti.minimize(w_trans * self.translational_cost + w_rot * self.rotation_cost + w_reg * self.regularization_cost + w_smooth * self.smooth_cost)

        # 求解器配置
        opts = {
            "ipopt": {
                "print_level": 0,
                "max_iter": 1000,
                "tol": 1e-4,
                "acceptable_tol": 1e-4,
            },
            "print_time": False,
            "calc_lam_p": False,
        }
        self.opti.solver("ipopt", opts)

        # 初始化状态
        self.init_qdata = pin.neutral(self.reduced_robot.model)
        self.last_qdata = pin.neutral(self.reduced_robot.model)

        # 平滑滤波器（使用4个历史数据）
        self.smooth_filter = WeightedMovingFilter(np.array([0.4, 0.3, 0.2, 0.1]), self.reduced_robot.model.nq)

        self.vis = None

        # 初始化可视化
        if self.Visualization:
            self._init_visualization()

    def _init_visualization(self):
        """初始化可视化"""
        self.vis = MeshcatVisualizer(self.reduced_robot.model, self.reduced_robot.collision_model, self.reduced_robot.visual_model)
        self.vis.initViewer(open=True)
        self.vis.loadViewerModel("pinocchio")

        # 显示坐标系
        self.vis.displayFrames(True, frame_ids=[self.endeff_id], axis_length=0.1, axis_width=5)
        self.vis.display(pin.neutral(self.reduced_robot.model))

        # # 添加目标坐标系显示
        # FRAME_AXIS_POSITIONS = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1]]).astype(np.float32).T

        # FRAME_AXIS_COLORS = np.array([[1, 0, 0], [1, 0.6, 0], [0, 1, 0], [0.6, 1, 0], [0, 0, 1], [0, 0.6, 1]]).astype(np.float32).T

        # axis_length = 0.15
        # axis_width = 8

        # self.vis.viewer["endeff_target"].set_object(
        #     mg.LineSegments(
        #         mg.PointsGeometry(
        #             position=axis_length * FRAME_AXIS_POSITIONS,
        #             color=FRAME_AXIS_COLORS,
        #         ),
        #         mg.LineBasicMaterial(
        #             linewidth=axis_width,
        #             vertexColors=True,
        #         ),
        #     )
        # )

    def solve_ik(self, target_pose, current_q=None, current_dq=None):
        """
        求解逆运动学

        Args:
            target_pose: 4x4齐次变换矩阵, 表示目标位姿
            current_q: 当前关节位置（用于初始化和平滑）
            current_dq: 当前关节速度（未使用, 保留接口）

        Returns:
            sol_q: 求解的关节位置
        """
        # 设置初始猜测
        if current_q is not None:
            self.init_qdata = current_q

        self.opti.set_initial(self.var_q, self.init_qdata)

        # 设置目标位姿参数
        self.opti.set_value(self.param_tf_endeff, target_pose)

        # 设置平滑参数
        self.opti.set_value(self.var_q_last, self.last_qdata)

        try:
            # 求解优化问题
            sol = self.opti.solve()
            sol_q = self.opti.value(self.var_q)

            # 应用平滑滤波
            self.smooth_filter.add_data(sol_q)
            sol_q = self.smooth_filter.filtered_data

            # 更新状态
            self.last_qdata = self.init_qdata
            self.init_qdata = sol_q

            # 可视化
            if self.Visualization:
                self.vis.viewer["endeff_target"].set_transform(target_pose)
                self.vis.display(sol_q)

            return sol_q

        except Exception as e:
            print(f"逆运动学求解失败: {e}")
            print("使用调试值...")

            # 获取调试值
            sol_q = self.opti.debug.value(self.var_q)
            self.smooth_filter.add_data(sol_q)
            sol_q = self.smooth_filter.filtered_data

            # 更新状态
            self.last_qdata = self.init_qdata
            self.init_qdata = sol_q

            # 可视化
            if self.Visualization:
                self.vis.viewer["endeff_target"].set_transform(target_pose)
                self.vis.display(sol_q)

            return current_q if current_q is not None else sol_q

    def compute_fk(self, q):
        """计算正向运动学"""
        pin.framesForwardKinematics(self.reduced_robot.model, self.reduced_robot.data, q)
        return self.reduced_robot.data.oMf[self.endeff_id].copy()

    def get_joint_limits(self):
        """获取关节限位"""
        return {"lower": self.reduced_robot.model.lowerPositionLimit, "upper": self.reduced_robot.model.upperPositionLimit}


# 测试代码
if __name__ == "__main__":
    # 设置URDF路径（需要根据实际情况修改）
    current_path = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join("./data/urdf/piper_no_gripper_description.urdf")  # 假设机械臂URDF文件名为robot_arm.urdf

    # 检查文件是否存在
    if not os.path.exists(urdf_path):
        print(f"URDF文件不存在: {urdf_path}")
        print("请确保robot_arm.urdf文件存在")
        sys.exit(1)

    # 创建机械臂IK求解器
    arm_ik = RobotArmIK(urdf_path, Visualization=True)

    # 打印机器人信息
    print(f"机器人自由度: {arm_ik.reduced_robot.model.nq}")
    print(f"关节限位: {arm_ik.get_joint_limits()}")

    # 初始位置
    target_pose = pin.SE3(pin.Quaternion(1, 0, 0, 0).toRotationMatrix(), np.array([0.4, 0.0, 0.3])).homogeneous  # 目标位置

    # 初始关节位置（中立位置）
    init_q = pin.neutral(arm_ik.reduced_robot.model)

    # 测试逆运动学求解
    print("\n求解逆运动学...")
    sol_q = arm_ik.solve_ik(target_pose, current_q=init_q)
    print(f"求解结果: {sol_q}")

    # 验证正向运动学
    fk_pose = arm_ik.compute_fk(sol_q)
    print(f"正向运动学验证: \n{fk_pose.homogeneous}")
    print(f"目标位姿: \n{target_pose}")

    # 简单轨迹测试
    print("\n开始轨迹测试...")
    user_input = input("按Enter开始轨迹测试, 或输入q退出: ")
    if user_input.lower() != "q":
        step = 0
        max_steps = 100
        radius = 0.1
        center = np.array([0.4, 0.0, 0.3])

        while True:
            # 生成圆形轨迹
            angle = 2 * np.pi * step / max_steps
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            z = center[2] + 0.05 * np.sin(2 * angle)

            # 更新目标位姿
            target_pose = pin.SE3(pin.Quaternion(1, 0, 0, 0).toRotationMatrix(), np.array([x, y, z])).homogeneous

            # 求解逆运动学
            sol_q = arm_ik.solve_ik(target_pose, current_q=sol_q)

            step += 1
            time.sleep(0.05)  # 控制循环速度

        print("轨迹测试完成")
