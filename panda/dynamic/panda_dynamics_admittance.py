"""
Panda机械臂导纳控制仿真程序

本程序实现了基于MuJoCo物理引擎的Panda机械臂导纳控制（Admittance Control）。
导纳控制是一种力/位置混合控制方法，当机械臂受到外力时，会根据导纳模型
调整期望位置，使机械臂能够顺应外力运动。

导纳模型方程：M_d·Δxdd + B_d·Δxd + K_d·(Δx - Δx_target) = F_ext - F_ref
其中：
    - M_d: 虚拟质量矩阵（决定惯性响应）
    - B_d: 虚拟阻尼矩阵（决定阻尼特性）
    - K_d: 虚拟刚度矩阵（决定弹性响应）
    - F_ext: 六维力/力矩传感器反馈（世界坐标系）
    - F_ref: 参考力/力矩
    - Δx: 相对初始末端位姿的柔顺偏移
    - Δx_target: 用户输入的累计增量位姿，作为导纳模型的新平衡点
"""

import argparse
from pathlib import Path
import select
import sys
import threading

import mujoco
import mujoco_viewer as mujoco_viewer
import numpy as np
from ik_dls import PinocchioSolver
import utils as utils
import lowpass_filter as lowpass_filter


class PandaEnv(mujoco_viewer.CustomViewer):
    """
    Panda机械臂导纳控制环境类

    继承自CustomViewer，实现MuJoCo仿真环境的可视化与控制逻辑
    """

    def __init__(self, scene_xml, arm_xml):
        """
        初始化Panda机械臂仿真环境

        Args:
            scene_xml: 场景XML文件路径（包含环境、物体等）
            arm_xml: 机械臂XML文件路径（包含机械臂模型）
        """
        # 调用父类构造函数，设置相机视角
        super().__init__(scene_xml, 3, azimuth=-45, elevation=-30)
        self.scene_xml = scene_xml
        self.arm_xml = arm_xml
        self._pose_command_lock = threading.Lock()
        self._pending_pose_increment = np.zeros(6)
        self._reset_pose_target_requested = False
        self.verbose_sensor_data = False

    def runBefore(self):
        """
        仿真循环开始前的初始化函数
        在run_loop()中会在主循环之前调用一次
        """
        # 获取初始关节位置（从keyframe中读取）
        self.initial_pos = self.model.key_qpos[0].copy()
        self.data.qpos[:] = self.initial_pos
        # 设置前7个关节的控制量为初始位置（Panda有7个关节）
        self.data.ctrl[:7] = self.initial_pos[:7]
        if self.model.nu > 7 and self.model.nkey > 0:
            self.data.ctrl[7] = self.model.key_ctrl[0, 7]

        # 末端执行器（End-Effector）的body名称
        self.ee_body_name = "ee_center_body"
        self.ee_ft_site_name = "ee_ft_site"
        self.force_sensor_name = "ee_force"
        self.torque_sensor_name = "ee_torque"

        # 六维力/力矩传感器由两个三维传感器组成。
        self.force_sensor_slice = self._get_sensor_slice(self.force_sensor_name, expected_dim=3)
        print("force_sensor_slice = ", self.force_sensor_slice)
        self.torque_sensor_slice = self._get_sensor_slice(self.torque_sensor_name, expected_dim=3)
        print("self.torque_sensor_slice = ", self.torque_sensor_slice)
        self.ee_ft_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.ee_ft_site_name)
        if self.ee_ft_site_id < 0:
            raise ValueError(f"MuJoCo模型中未找到末端传感器site: {self.ee_ft_site_name}")

        # 初始化逆运动学求解器（使用Pinocchio库）
        source_root = Path(__file__).resolve().parents[4]
        panda_urdf = source_root / "description/panda_description/urdf/panda.urdf"
        self.arm = PinocchioSolver(str(panda_urdf))
        self.arm.ee_joint_id = self.arm.model.getJointId("panda_joint7")
        self.arm.ee_joint_name = "panda_link7"
        if self.arm.ee_joint_id <= 0:
            raise ValueError("Panda URDF中未找到panda_joint7")

        # PinocchioSolver以link7为求解末端；控制目标位于TCP，因此保存二者
        # 的固定变换，IK前把TCP目标换算为link7目标。
        mujoco.mj_forward(self.model, self.data)
        world_from_link7 = self._get_body_transform("link7")
        world_from_ee = self._get_body_transform(self.ee_body_name)
        link7_from_ee = np.linalg.inv(world_from_link7) @ world_from_ee
        self.ee_from_link7 = np.linalg.inv(link7_from_ee)

        # 保存上一次的关节角度（用于逆运动学求解的初始猜测）
        self.last_dof = self.data.qpos.copy()
        # 设置仿真时间步长为1ms（0.001秒）
        self.setTimestep(0.001)
        # 末端执行器期望速度的增量（6维：x, y, z, roll, pitch, yaw）
        self.delta_d_ee_des = np.zeros(6)
        # 末端执行器期望位置的增量（6维：x, y, z, roll, pitch, yaw）
        self.delta_ee_des = np.zeros(6)
        # 增量位姿命令对应的导纳平衡点，顺序为[x, y, z, roll, pitch, yaw]。
        self.target_delta_ee = np.zeros(6)
        self.pose_target_active = False
        self.position_target_tolerance = 1e-3
        self.orientation_target_tolerance = 1e-2
        self.target_velocity_tolerance = 1e-2
        # 前1秒保持初始位置，让位置内环稳定；最后0.25秒用于F/T零偏标定。
        self.first_goto_initial_pos_cnt = 1000
        self.wrench_calibration_steps = 250

        # LowPassOnlineFilter的第二个参数是时间常数tau，而不是截止频率。
        self.vel_filter = lowpass_filter.LowPassOnlineFilter(6, 0.1, self.model.opt.timestep)
        self.wrench_filter = lowpass_filter.LowPassOnlineFilter(6, 0.02, self.model.opt.timestep)

        # 传感器原始输出是“父体作用于子体的约束反力”。标定后取负号，
        # 得到外界作用于末端的等效力/力矩。
        self.wrench_bias_sum = np.zeros(6)
        self.wrench_bias = np.zeros(6)
        self.wrench_bias_samples = 0
        self.measured_wrench = np.zeros(6)
        self.reference_wrench = np.zeros(6)

        # 平移轴与旋转轴的参数量纲不同：
        # 平移为 kg、N·s/m、N/m；旋转为 kg·m²、N·m·s/rad、N·m/rad。
        self.M_d = np.diag([10.0, 10.0, 10.0, 0.5, 0.5, 0.5])
        self.B_d = np.diag([45.0, 45.0, 45.0, 4.5, 4.5, 4.5])
        self.K_d = np.diag([50.0, 50.0, 50.0, 10.0, 10.0, 10.0])

        # 防止接触冲击或IK异常造成指令突跳。
        self.wrench_limit = np.array([50.0, 50.0, 50.0, 5.0, 5.0, 5.0])
        self.velocity_limit = np.array([0.30, 0.30, 0.30, 0.50, 0.50, 0.50])
        self.displacement_limit = np.array([0.15, 0.15, 0.15, 0.35, 0.35, 0.35])
        self.control_step = 0
        self.simulation_step = 0
        self.terminal_input_enabled = True

        if sys.stdin.isatty():
            print(
                "输入增量位姿: dx dy dz droll dpitch dyaw "
                "（位置m，姿态rad，世界坐标系）；输入 reset 返回初始平衡点。"
            )

        # 以下为注释掉的实时绘图代码（用于调试和可视化）
        # import src.matplot as mp
        # self.plot_manager = mp.MultiChartRealTimePlotManager()
        # self.plot_manager.addNewFigurePlotter("vel.x", "vel.x", row=0, col=0)
        # self.plot_manager.addNewFigurePlotter("delta.x", title="delta.x", row=1, col=0)
        # self.plot_manager.addNewFigurePlotter("delta.y", title="delta.y", row=2, col=0)
        # self.plot_manager.addNewFigurePlotter("delta.z", title="delta.z", row=3, col=0)

        # 逆运动学停止标志：当遇到边界或求解失败时停止更新
        self.ik_stop = False

    def add_pose_increment(self, delta_pose):
        """提交一个六维增量位姿命令，可由终端、ROS回调或其他代码调用。"""
        delta_pose = np.asarray(delta_pose, dtype=float)
        if delta_pose.shape != (6,):
            raise ValueError(
                "增量位姿必须是6维：[dx, dy, dz, droll, dpitch, dyaw]"
            )
        if not np.all(np.isfinite(delta_pose)):
            raise ValueError("增量位姿不能包含NaN或无穷大")

        with self._pose_command_lock:
            self._pending_pose_increment += delta_pose

    def reset_pose_target(self):
        """请求把导纳平衡点重置到启动时的末端位姿。"""
        with self._pose_command_lock:
            self._pending_pose_increment.fill(0.0)
            self._reset_pose_target_requested = True

    def _poll_terminal_pose_command(self):
        """非阻塞读取终端中的六维增量位姿。"""
        if not self.terminal_input_enabled:
            return

        try:
            readable, _, _ = select.select([sys.stdin], [], [], 0.0)
        except (OSError, ValueError):
            self.terminal_input_enabled = False
            return

        if not readable:
            return

        line = sys.stdin.readline()
        if line == "":
            self.terminal_input_enabled = False
            return

        command = line.strip()
        if not command:
            return
        if command.lower() == "reset":
            self.reset_pose_target()
            return

        try:
            values = [float(value) for value in command.replace(",", " ").split()]
            self.add_pose_increment(values)
        except ValueError as exc:
            print(
                "无效增量位姿。请输入6个数："
                "dx dy dz droll dpitch dyaw，或输入 reset。"
                f" 错误：{exc}"
            )

    def _consume_pose_command(self):
        """消费待处理输入，并更新受限的导纳平衡点。"""
        with self._pose_command_lock:
            pose_increment = self._pending_pose_increment.copy()
            self._pending_pose_increment.fill(0.0)
            reset_requested = self._reset_pose_target_requested
            self._reset_pose_target_requested = False

        if reset_requested:
            requested_target = np.zeros(6)
        elif np.any(np.abs(pose_increment) > 1e-12):
            requested_target = self.target_delta_ee + pose_increment
        else:
            return False

        self.target_delta_ee = np.clip(
            requested_target, -self.displacement_limit, self.displacement_limit
        )
        self.pose_target_active = True
        self.ik_stop = False
        print(
            "收到增量位姿命令，新的导纳目标 [m, rad]：",
            np.round(self.target_delta_ee, 4),
        )
        return True

    def _update_pose_target_status(self):
        """检测虚拟导纳状态是否已经收敛到输入目标。"""
        if not self.pose_target_active:
            return

        pose_error = self.target_delta_ee - self.delta_ee_des
        position_reached = np.linalg.norm(pose_error[:3]) <= self.position_target_tolerance
        orientation_reached = (
            np.linalg.norm(pose_error[3:]) <= self.orientation_target_tolerance
        )
        velocity_reached = (
            np.linalg.norm(self.delta_d_ee_des) <= self.target_velocity_tolerance
        )
        if position_reached and orientation_reached and velocity_reached:
            self.pose_target_active = False
            print(
                "导纳目标已到达，当前偏移 [m, rad]：",
                np.round(self.delta_ee_des, 4),
            )

    def _get_sensor_slice(self, sensor_name, expected_dim):
        """返回指定MuJoCo传感器在data.sensordata中的切片。"""
        sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
        if sensor_id < 0:
            raise ValueError(f"MuJoCo模型中未找到传感器: {sensor_name}，" "请确认panda_pos.xml已声明<sensor>。")

        sensor_dim = int(self.model.sensor_dim[sensor_id])
        if sensor_dim != expected_dim:
            raise ValueError(f"传感器{sensor_name}维度为{sensor_dim}，期望{expected_dim}。")

        sensor_adr = int(self.model.sensor_adr[sensor_id])
        return slice(sensor_adr, sensor_adr + sensor_dim)

    def _get_body_transform(self, body_name):
        """返回MuJoCo body在世界坐标系中的4x4齐次变换。"""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"MuJoCo模型中未找到body: {body_name}")

        transform = np.eye(4)
        transform[:3, :3] = self.data.xmat[body_id].reshape(3, 3)
        transform[:3, 3] = self.data.xpos[body_id]
        return transform

    def _read_raw_wrench_world(self):
        """读取传感器反力，并从site局部坐标系旋转到世界坐标系。"""
        force_local = self.data.sensordata[self.force_sensor_slice].copy()
        torque_local = self.data.sensordata[self.torque_sensor_slice].copy()
        rotation_world_from_site = self.data.site_xmat[self.ee_ft_site_id].reshape(3, 3)

        if self.verbose_sensor_data:
            print("wrench:", self.data.sensordata[:6])

        force_world = rotation_world_from_site @ force_local
        torque_world = rotation_world_from_site @ torque_local
        return np.concatenate((force_world, torque_world))

    def _read_external_wrench_world(self):
        """返回经零偏补偿、符号修正和低通滤波后的末端外力。"""
        raw_wrench = self._read_raw_wrench_world()
        external_wrench = -(raw_wrench - self.wrench_bias)
        external_wrench = np.clip(external_wrench, -self.wrench_limit, self.wrench_limit)
        return self.wrench_filter.update(external_wrench)

    def runFunc(self):
        """
        每个仿真步骤执行的主控制函数
        实现导纳控制的核心逻辑：检测外力 -> 计算期望位置 -> 逆运动学求解 -> 更新关节角度
        """
        self.simulation_step += 1
        if self.simulation_step % 20 == 0:
            self._poll_terminal_pose_command()

        # ========== 第一阶段：初始化与传感器标定 ==========
        if self.first_goto_initial_pos_cnt > 0:
            # 递减计数器
            self.first_goto_initial_pos_cnt -= 1
            # 将关节控制量设置为初始位置
            self.data.ctrl[:7] = self.initial_pos[:7]
            # 获取当前末端执行器位姿（6维：x, y, z, roll, pitch, yaw）
            self.ee_pos = self.getBodyPoseEulerByName(self.ee_body_name)
            # 初始化期望位置为当前位置
            self.desired_pos = self.ee_pos.copy()
            # 保存上一次的末端执行器位置（用于计算速度）
            self.last_ee_pos = self.ee_pos.copy()
            # 保存起始位置（用于计算位置增量）
            self.start_ee_pos = self.ee_pos.copy()

            # 只使用稳定后的样本，避免把位置内环启动瞬态计入零偏。
            if self.first_goto_initial_pos_cnt < self.wrench_calibration_steps:
                self.wrench_bias_sum += self._read_raw_wrench_world()
                self.wrench_bias_samples += 1
                self.wrench_bias = self.wrench_bias_sum / self.wrench_bias_samples
        else:
            # ========== 第二阶段：导纳控制阶段 ==========

            # 检测新的增量位姿输入，并将其累加到导纳平衡点。
            self._consume_pose_command()

            # --- 1. 获取当前末端执行器状态 ---
            # 获取当前末端执行器位姿
            self.now_ee_pos = self.getBodyPoseEulerByName(self.ee_body_name)
            # 通过数值微分计算当前速度：v = (p_now - p_last) / dt
            self.now_ee_vel = (self.now_ee_pos - self.last_ee_pos) / self.model.opt.timestep
            # 更新上一次位置
            self.last_ee_pos = self.now_ee_pos.copy()
            # 使用低通滤波器平滑速度信号（减少噪声）
            self.now_ee_vel_filter = self.vel_filter.update(self.now_ee_vel)
            # --- 2. 力检测与计算 ---
            # 六维顺序：[Fx, Fy, Fz, Mx, My, Mz]，均已转换到世界坐标系。
            self.measured_wrench = self._read_external_wrench_world()
            F_e = self.measured_wrench - self.reference_wrench

            # --- 3. 导纳控制方程求解 ---
            # M_d·Δxdd + B_d·Δxd + K_d·(Δx - Δx_target) = F_ext - F_ref
            pose_target_error = self.delta_ee_des - self.target_delta_ee
            dd_ee = np.linalg.solve(
                self.M_d,
                F_e
                - self.B_d @ self.delta_d_ee_des
                - self.K_d @ pose_target_error,
            )

            # --- 4. 积分计算期望速度和位置 ---
            # 通过积分加速度得到速度增量：dv = a * dt
            self.delta_d_ee_des += dd_ee * self.model.opt.timestep
            self.delta_d_ee_des = np.clip(self.delta_d_ee_des, -self.velocity_limit, self.velocity_limit)
            # 通过积分速度得到位置增量：dp = v * dt
            self.delta_ee_des += self.delta_d_ee_des * self.model.opt.timestep
            self.delta_ee_des = np.clip(self.delta_ee_des, -self.displacement_limit, self.displacement_limit)
            self._update_pose_target_status()
            # 计算新的期望位置：起始位置 + 位置增量
            self.desired_pos = self.start_ee_pos + self.delta_ee_des

            # --- 5. 逆运动学求解 ---
            # 将期望位姿（位置+欧拉角）转换为齐次变换矩阵（4x4）
            tf = utils.transform2mat(self.desired_pos[0], self.desired_pos[1], self.desired_pos[2], self.desired_pos[3], self.desired_pos[4], self.desired_pos[5])  # x, y, z  # roll, pitch, yaw
            tf_link7 = tf @ self.ee_from_link7
            # 使用逆运动学求解器计算关节角度
            # tf[:3, 3]: 位置向量（3x1）
            # tf[:3, :3]: 旋转矩阵（3x3）
            # self.last_dof[:9]: 上一次的关节角度（作为初始猜测，提高求解效率）
            flag, self.dof, _ = self.arm.ik(tf_link7[:3, 3], tf_link7[:3, :3], self.last_dof[: self.arm.nq])

            # --- 6. 边界检查和停止条件 ---
            # 如果z轴位置过小（接近0）或逆运动学求解失败，停止更新
            if self.desired_pos[2] < 0.001 or not flag:
                self.ik_stop = True

            # --- 7. 更新关节位置执行器目标 ---
            if not self.ik_stop:
                # 保存当前关节角度（用于下一次逆运动学求解的初始猜测）
                self.last_dof = self.dof
                # 通过位置执行器跟踪IK结果，不直接覆写qpos，保留真实动力学和接触响应。
                self.data.ctrl[:7] = self.dof[:7]

                self.control_step += 1
                if self.control_step % 100 == 0:
                    # print(
                    #     "EE wrench world [N, Nm]:",
                    #     np.round(self.measured_wrench, 3),
                    # )
                    pass

                # 以下为注释掉的实时绘图代码（用于调试和可视化）
                # self.plot_manager.updateDataToPlotter("vel.x", "now_ee_vel.x", self.now_ee_vel[0])
                # self.plot_manager.updateDataToPlotter("vel.x", "now_ee_velfilter.x", self.now_ee_vel_filter[0])
                # self.plot_manager.updateDataToPlotter("delta.x", "delta.x", self.desired_pos[0])
                # self.plot_manager.updateDataToPlotter("delta.y", "delta.y", self.desired_pos[1])
                # self.plot_manager.updateDataToPlotter("delta.z", "delta.z", self.desired_pos[2])
                # print("vel.x", "now_ee_vel.x", self.now_ee_vel[0])
                # print("vel.y", "now_ee_vel.y", self.now_ee_vel[1])
            else:
                # 如果停止标志为True，保持上一次的关节角度不变
                # print("vel.x", "now_ee_vel.x", self.now_ee_vel[0])
                # print("vel.y", "now_ee_vel.y", self.now_ee_vel[1])
                self.data.ctrl[:7] = self.last_dof[:7]


if __name__ == "__main__":
    """
    主函数：创建仿真环境并启动仿真循环

    使用说明：
    1. 加载场景和机械臂模型
    2. 创建PandaEnv环境实例
    3. 启动仿真循环（run_loop会自动调用runBefore和runFunc）
    """
    parser = argparse.ArgumentParser(description="Panda六维导纳控制仿真")
    parser.add_argument(
        "--delta-pose",
        nargs=6,
        type=float,
        metavar=("DX", "DY", "DZ", "DROLL", "DPITCH", "DYAW"),
        help="启动后执行一次六维增量位姿命令，位置单位m、姿态单位rad",
    )
    parser.add_argument(
        "--verbose-raw-wrench",
        action="store_true",
        help="每个控制周期打印MuJoCo原始六维传感器数据",
    )
    args = parser.parse_args()

    # 场景XML文件路径（包含环境、桌面、物体等）
    mujoco_root = Path(__file__).resolve().parents[2]
    model_dir = mujoco_root / "model/franka_emika_panda"
    SCENE_XML = str(model_dir / "scene_pos.xml")
    # 机械臂XML文件路径（包含Panda机械臂模型）
    ARM_XML = str(model_dir / "panda_pos.xml")

    # 创建Panda机械臂导纳控制环境
    env = PandaEnv(SCENE_XML, ARM_XML)
    env.verbose_sensor_data = args.verbose_raw_wrench
    if args.delta_pose is not None:
        env.add_pose_increment(args.delta_pose)

    # 启动仿真循环
    # run_loop()会：
    #   1. 调用runBefore()进行初始化
    #   2. 循环执行：前向动力学 -> runFunc() -> 步进仿真 -> 同步显示
    env.run_loop()
