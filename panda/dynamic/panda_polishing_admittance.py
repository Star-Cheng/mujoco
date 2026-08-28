"""Panda机械臂恒力打磨任务仿真。

控制结构：
    - 切向：名义位置轨迹，沿工件表面往复运动。
    - 法向：力误差导纳，输入 F_ref - F_measured，输出法向位置修正。
    - 执行：名义轨迹 + 导纳修正 -> 末端位姿 -> IK -> 关节位置内环。

法向导纳方程：
    M_n * d2(delta_n) + B_n * d(delta_n) + K_n * delta_n
        = F_ref - F_measured

本示例把工件局部 +Z 定义为“压入工件”的正方向。
"""

import argparse
from enum import Enum, auto
from pathlib import Path

import mujoco
import numpy as np

import lowpass_filter
import mujoco_viewer
from ik_dls import PinocchioSolver
import utils


class PolishingState(Enum):
    SETTLE = auto()
    APPROACH = auto()
    FORCE_RAMP = auto()
    POLISH = auto()
    RETRACT = auto()
    DONE = auto()
    FAULT = auto()


class PandaPolishingEnv(mujoco_viewer.CustomViewer):
    """Panda平面恒力打磨仿真环境。"""

    def __init__(
        self,
        scene_xml,
        arm_xml,
        target_force=8.0,
        trajectory="raster",
        stroke=0.02,
        path_frequency=0.03,
        raster_width=0.04,
        raster_height=0.03,
        raster_rows=4,
        path_speed=0.01,
        polish_duration=22.0,
    ):
        super().__init__(scene_xml, 2.2, azimuth=135, elevation=-18)
        self.scene_xml = scene_xml
        self.arm_xml = arm_xml
        self.target_force = float(target_force)
        self.trajectory = str(trajectory)
        self.stroke = float(stroke)
        self.path_frequency = float(path_frequency)
        self.raster_width = float(raster_width)
        self.raster_height = float(raster_height)
        self.raster_rows = int(raster_rows)
        self.path_speed = float(path_speed)
        self.polish_duration = float(polish_duration)

        if self.target_force <= 0:
            raise ValueError("target_force必须大于0")
        if self.trajectory not in ("raster", "line"):
            raise ValueError("trajectory必须是raster或line")
        if not 0 < self.stroke <= 0.10:
            raise ValueError("stroke必须在(0, 0.10] m内")
        if self.path_frequency <= 0:
            raise ValueError("path_frequency必须大于0")
        if not 0 < self.raster_width <= 0.12:
            raise ValueError("raster_width必须在(0, 0.12] m内")
        if not 0 < self.raster_height <= 0.07:
            raise ValueError("raster_height必须在(0, 0.07] m内")
        if self.raster_rows < 2:
            raise ValueError("raster_rows必须不小于2")
        if not 0 < self.path_speed <= 0.03:
            raise ValueError("path_speed必须在(0, 0.03] m/s内")
        if self.polish_duration <= 0:
            raise ValueError("polish_duration必须大于0")

    def runBefore(self):
        self.setTimestep(0.001)
        self.dt = float(self.model.opt.timestep)

        key_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, "polishing_home"
        )
        if key_id < 0:
            raise ValueError("scene_polishing.xml中未找到polishing_home keyframe")
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        # 使用keyframe关节位置作为位置内环目标，不使用其中的旧ctrl值。
        self.data.ctrl[:7] = self.data.qpos[:7]
        if self.model.nu > 7:
            self.data.ctrl[7] = 255.0
        # 打磨的笛卡尔外环要求关节位置内环明显更快、更硬。
        self.joint_kp = 4000.0
        self.joint_kv = 200.0
        self.model.actuator_gainprm[:7, 0] = self.joint_kp
        self.model.actuator_biasprm[:7, 1] = -self.joint_kp
        self.model.actuator_biasprm[:7, 2] = -self.joint_kv
        mujoco.mj_forward(self.model, self.data)

        self.ee_body_name = "ee_center_body"
        self.force_sensor_slice = self._get_sensor_slice("ee_force", 3)
        self.torque_sensor_slice = self._get_sensor_slice("ee_torque", 3)
        self.ee_ft_site_id = self._require_id(
            mujoco.mjtObj.mjOBJ_SITE, "ee_ft_site"
        )
        self.pad_geom_id = self._require_id(
            mujoco.mjtObj.mjOBJ_GEOM, "polishing_pad"
        )
        self.workpiece_geom_id = self._require_id(
            mujoco.mjtObj.mjOBJ_GEOM, "workpiece_geom"
        )
        self.workpiece_body_id = self._require_id(
            mujoco.mjtObj.mjOBJ_BODY, "workpiece"
        )
        self.ee_body_id = self._require_id(
            mujoco.mjtObj.mjOBJ_BODY, self.ee_body_name
        )

        # Pinocchio以link7为末端，MuJoCo控制的是TCP；保存两者固定变换。
        source_root = Path(__file__).resolve().parents[4]
        panda_urdf = source_root / "description/panda_description/urdf/panda.urdf"
        self.arm = PinocchioSolver(str(panda_urdf))
        self.arm.ee_joint_id = self.arm.model.getJointId("panda_joint7")
        self.arm.ee_joint_name = "panda_link7"
        if self.arm.ee_joint_id <= 0:
            raise ValueError("Panda URDF中未找到panda_joint7")

        world_from_link7 = self._get_body_transform("link7")
        world_from_ee = self._get_body_transform(self.ee_body_name)
        link7_from_ee = np.linalg.inv(world_from_link7) @ world_from_ee
        self.ee_from_link7 = np.linalg.inv(link7_from_ee)
        self.last_dof = self.data.qpos[: self.arm.nq].copy()
        self.joint_control_bias = np.zeros(7)

        # 工件坐标系：X/Y在表面内，+Z指向工件内部。
        surface_rotation = self.data.xmat[self.workpiece_body_id].reshape(3, 3)
        self.path_axis_world = surface_rotation[:, 0].copy()
        self.cross_axis_world = surface_rotation[:, 1].copy()
        self.normal_axis_world = surface_rotation[:, 2].copy()

        self.state = PolishingState.SETTLE
        self.state_start_time = float(self.data.time)
        self.simulation_step = 0
        self.settle_steps = int(1.0 / self.dt)
        self.wrench_calibration_steps = int(0.25 / self.dt)

        self.wrench_bias_sum = np.zeros(6)
        self.wrench_bias = np.zeros(6)
        self.wrench_bias_samples = 0
        self.wrench_filter = lowpass_filter.LowPassOnlineFilter(6, 0.035, self.dt)
        self.wrench_limit = np.array([60.0, 60.0, 60.0, 6.0, 6.0, 6.0])
        self.measured_wrench = np.zeros(6)
        self.measured_normal_force = 0.0
        self.reference_normal_force = 0.0

        # 法向导纳参数。K_n=0使稳态条件直接对应力误差趋零。
        self.normal_mass = 8.0
        self.normal_damping = 900.0
        self.normal_stiffness = 0.0
        self.normal_offset = 0.0
        self.normal_velocity = 0.0
        self.normal_velocity_limit = 0.008
        self.normal_offset_limit = 0.020

        self.approach_speed = 0.012
        self.max_approach_distance = 0.055
        self.contact_force_threshold = 0.6
        self.force_ramp_duration = 2.0
        self.max_safe_force = max(40.0, 4.0 * self.target_force)
        self.retract_speed = 0.035
        self.retract_distance = 0.06
        self.initial_surface_gap = 0.028
        self.pad_front_from_tcp = 0.019
        self.workpiece_half_thickness = 0.020

        self.start_pose = self.getBodyPoseEulerByName(self.ee_body_name)
        self.nominal_pose = self.start_pose.copy()
        self.desired_pose = self.start_pose.copy()
        self.contact_pose = None
        self.contact_force_at_transition = 0.0
        self.polish_start_time = None
        self.retract_start_pose = None
        self.path_offset = np.zeros(2)
        self.ik_failed = False
        self.polish_force_samples = []

        if self.trajectory == "raster":
            trajectory_description = (
                f"栅格 {self.raster_width:.3f} x {self.raster_height:.3f} m，"
                f"{self.raster_rows}行"
            )
        else:
            trajectory_description = f"直线单边行程 {self.stroke:.3f} m"
        print(
            "打磨仿真启动：目标法向力 "
            f"{self.target_force:.2f} N，轨迹 {trajectory_description}"
        )

    def _require_id(self, object_type, name):
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"MuJoCo模型中未找到{name}")
        return object_id

    def _get_sensor_slice(self, sensor_name, expected_dim):
        sensor_id = self._require_id(mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
        sensor_dim = int(self.model.sensor_dim[sensor_id])
        if sensor_dim != expected_dim:
            raise ValueError(
                f"传感器{sensor_name}维度为{sensor_dim}，期望{expected_dim}"
            )
        sensor_adr = int(self.model.sensor_adr[sensor_id])
        return slice(sensor_adr, sensor_adr + sensor_dim)

    def _get_body_transform(self, body_name):
        body_id = self._require_id(mujoco.mjtObj.mjOBJ_BODY, body_name)
        transform = np.eye(4)
        transform[:3, :3] = self.data.xmat[body_id].reshape(3, 3)
        transform[:3, 3] = self.data.xpos[body_id]
        return transform

    def _read_raw_wrench_world(self):
        force_local = self.data.sensordata[self.force_sensor_slice].copy()
        torque_local = self.data.sensordata[self.torque_sensor_slice].copy()
        rotation_world_from_site = self.data.site_xmat[self.ee_ft_site_id].reshape(
            3, 3
        )
        return np.concatenate(
            (
                rotation_world_from_site @ force_local,
                rotation_world_from_site @ torque_local,
            )
        )

    def _read_external_wrench_world(self):
        # MuJoCo传感器指向为子体到父体；取反后表示外界作用于工具。
        external_wrench = -(self._read_raw_wrench_world() - self.wrench_bias)
        external_wrench = np.clip(
            external_wrench, -self.wrench_limit, self.wrench_limit
        )
        return self.wrench_filter.update(external_wrench)

    def _pad_touches_workpiece(self):
        target_pair = {self.pad_geom_id, self.workpiece_geom_id}
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            if {int(contact.geom1), int(contact.geom2)} == target_pair:
                return True
        return False

    def _set_state(self, new_state, message):
        self.state = new_state
        self.state_start_time = float(self.data.time)
        print(f"[{self.data.time:7.3f}s] {new_state.name}: {message}")

    def _align_workpiece_to_settled_tool(self):
        """在无接触状态下把工件表面与当前工具端面精确对齐。"""
        tool_rotation = self.data.xmat[self.ee_body_id].reshape(3, 3).copy()
        tool_quaternion = self.data.xquat[self.ee_body_id].copy()
        tool_position = self.data.xpos[self.ee_body_id].copy()

        self.path_axis_world = tool_rotation[:, 0]
        self.cross_axis_world = tool_rotation[:, 1]
        self.normal_axis_world = tool_rotation[:, 2]
        center_distance = (
            self.pad_front_from_tcp
            + self.initial_surface_gap
            + self.workpiece_half_thickness
        )
        self.model.body_pos[self.workpiece_body_id] = (
            tool_position + self.normal_axis_world * center_distance
        )
        self.model.body_quat[self.workpiece_body_id] = tool_quaternion
        mujoco.mj_forward(self.model, self.data)

    def _update_normal_admittance(self):
        force_error = self.reference_normal_force - self.measured_normal_force
        normal_acceleration = (
            force_error
            - self.normal_damping * self.normal_velocity
            - self.normal_stiffness * self.normal_offset
        ) / self.normal_mass

        self.normal_velocity += normal_acceleration * self.dt
        self.normal_velocity = float(
            np.clip(
                self.normal_velocity,
                -self.normal_velocity_limit,
                self.normal_velocity_limit,
            )
        )
        self.normal_offset += self.normal_velocity * self.dt
        clipped_offset = float(
            np.clip(
                self.normal_offset,
                -self.normal_offset_limit,
                self.normal_offset_limit,
            )
        )
        if clipped_offset != self.normal_offset:
            self.normal_velocity = 0.0
        self.normal_offset = clipped_offset

    def _command_pose(self, desired_pose):
        target_transform = utils.transform2mat(*desired_pose)
        target_link7 = target_transform @ self.ee_from_link7
        flag, dof, _ = self.arm.ik(
            target_link7[:3, 3],
            target_link7[:3, :3],
            self.last_dof[: self.arm.nq],
        )
        if not flag or not np.all(np.isfinite(dof[:7])):
            self.ik_failed = True
            return False

        self.last_dof = dof.copy()
        biased_joint_target = dof[:7] + self.joint_control_bias
        self.data.ctrl[:7] = np.clip(
            biased_joint_target,
            self.model.actuator_ctrlrange[:7, 0],
            self.model.actuator_ctrlrange[:7, 1],
        )
        return True

    def _hold_initial_and_calibrate(self):
        self.start_pose = self.getBodyPoseEulerByName(self.ee_body_name)
        self.nominal_pose = self.start_pose.copy()
        self.desired_pose = self.start_pose.copy()

        remaining_steps = self.settle_steps - self.simulation_step
        if remaining_steps < self.wrench_calibration_steps:
            self.wrench_bias_sum += self._read_raw_wrench_world()
            self.wrench_bias_samples += 1
            self.wrench_bias = self.wrench_bias_sum / self.wrench_bias_samples

        if self.simulation_step >= self.settle_steps:
            # 位置执行器需要依靠静态位置误差抵消重力和工具载荷。保存这部分
            # 预载偏置，否则第一次IK指令会消掉预载并造成末端明显下沉。
            self.last_dof = self.data.qpos[: self.arm.nq].copy()
            self.joint_control_bias = self.data.ctrl[:7] - self.data.qpos[:7]
            self._align_workpiece_to_settled_tool()
            self._set_state(
                PolishingState.APPROACH,
                "F/T零偏标定完成，开始接近工件",
            )

    def _run_approach(self):
        elapsed = float(self.data.time - self.state_start_time)
        approach_distance = min(
            self.approach_speed * elapsed, self.max_approach_distance
        )
        self.nominal_pose = self.start_pose.copy()
        self.nominal_pose[:3] += self.normal_axis_world * approach_distance
        self.desired_pose = self.nominal_pose.copy()

        if (
            self._pad_touches_workpiece()
            and self.measured_normal_force >= self.contact_force_threshold
        ):
            self.contact_pose = self.desired_pose.copy()
            self.contact_force_at_transition = self.measured_normal_force
            self.normal_offset = 0.0
            self.normal_velocity = 0.0
            self._set_state(
                PolishingState.FORCE_RAMP,
                f"检测到接触 {self.measured_normal_force:.2f} N，开始力渐增",
            )
        elif approach_distance >= self.max_approach_distance:
            self._set_state(
                PolishingState.FAULT,
                "超过最大接近距离仍未检测到接触",
            )

    def _run_force_ramp(self):
        elapsed = float(self.data.time - self.state_start_time)
        ramp_ratio = float(np.clip(elapsed / self.force_ramp_duration, 0.0, 1.0))
        smooth_ratio = ramp_ratio * ramp_ratio * (3.0 - 2.0 * ramp_ratio)
        self.reference_normal_force = (
            self.contact_force_at_transition
            + smooth_ratio
            * (self.target_force - self.contact_force_at_transition)
        )

        self.nominal_pose = self.contact_pose.copy()
        self._update_normal_admittance()
        self.desired_pose = self.nominal_pose.copy()
        self.desired_pose[:3] += self.normal_axis_world * self.normal_offset

        if ramp_ratio >= 1.0:
            self.reference_normal_force = self.target_force
            self.polish_start_time = float(self.data.time)
            self._set_state(
                PolishingState.POLISH,
                "目标力已建立，开始沿表面执行打磨轨迹",
            )

    @staticmethod
    def _smoothstep(progress):
        """三次平滑插值，使每个轨迹段在端点处速度连续归零。"""
        progress = float(np.clip(progress, 0.0, 1.0))
        return progress * progress * (3.0 - 2.0 * progress)

    def _raster_trajectory_offset(self, elapsed):
        """返回蛇形栅格轨迹在工件X/Y方向上的二维偏移。"""
        half_width = 0.5 * self.raster_width
        half_height = 0.5 * self.raster_height
        row_step = self.raster_height / (self.raster_rows - 1)
        pass_duration = self.raster_width / self.path_speed
        cross_duration = max(0.4, row_step / self.path_speed)

        first_point = np.array([-half_width, -half_height])
        entry_duration = max(0.5, np.linalg.norm(first_point) / self.path_speed)
        if elapsed < entry_duration:
            progress = self._smoothstep(elapsed / entry_duration)
            return first_point * progress

        raster_elapsed = elapsed - entry_duration
        cycle_duration = (
            self.raster_rows * pass_duration
            + (self.raster_rows - 1) * cross_duration
        )
        cycle_index = int(raster_elapsed // cycle_duration)
        local_time = raster_elapsed - cycle_index * cycle_duration
        forward_rows = cycle_index % 2 == 0
        row_order = (
            list(range(self.raster_rows))
            if forward_rows
            else list(reversed(range(self.raster_rows)))
        )

        # 偶数/奇数行数都保持相邻cycle首尾连续。
        reverse_start_sign = -1.0 if self.raster_rows % 2 == 0 else 1.0
        cycle_start_sign = -1.0 if forward_rows else reverse_start_sign

        for order_index, row_index in enumerate(row_order):
            row_y = -half_height + row_index * row_step
            start_sign = cycle_start_sign * ((-1.0) ** order_index)
            start_x = start_sign * half_width
            end_x = -start_x

            if local_time <= pass_duration:
                progress = self._smoothstep(local_time / pass_duration)
                x_offset = start_x + (end_x - start_x) * progress
                return np.array([x_offset, row_y])
            local_time -= pass_duration

            if order_index == self.raster_rows - 1:
                return np.array([end_x, row_y])

            next_row = row_order[order_index + 1]
            next_y = -half_height + next_row * row_step
            if local_time <= cross_duration:
                progress = self._smoothstep(local_time / cross_duration)
                y_offset = row_y + (next_y - row_y) * progress
                return np.array([end_x, y_offset])
            local_time -= cross_duration

        return np.array([0.0, 0.0])

    def _trajectory_offset(self, elapsed):
        if self.trajectory == "line":
            phase = 2.0 * np.pi * self.path_frequency * elapsed
            return np.array([self.stroke * np.sin(phase), 0.0])
        return self._raster_trajectory_offset(elapsed)

    def _run_polish(self):
        elapsed = float(self.data.time - self.polish_start_time)
        self.reference_normal_force = self.target_force
        self.polish_force_samples.append(self.measured_normal_force)
        self.path_offset = self._trajectory_offset(elapsed)

        self.nominal_pose = self.contact_pose.copy()
        self.nominal_pose[:3] += self.path_axis_world * self.path_offset[0]
        self.nominal_pose[:3] += self.cross_axis_world * self.path_offset[1]
        self._update_normal_admittance()
        self.desired_pose = self.nominal_pose.copy()
        self.desired_pose[:3] += self.normal_axis_world * self.normal_offset

        if elapsed >= self.polish_duration:
            self.retract_start_pose = self.desired_pose.copy()
            self.reference_normal_force = 0.0
            self._set_state(
                PolishingState.RETRACT,
                "打磨时间完成，撤离工件",
            )

    def _run_retract(self):
        elapsed = float(self.data.time - self.state_start_time)
        retract_distance = min(
            self.retract_speed * elapsed, self.retract_distance
        )
        self.desired_pose = self.retract_start_pose.copy()
        self.desired_pose[:3] -= self.normal_axis_world * retract_distance
        if retract_distance >= self.retract_distance:
            self._set_state(PolishingState.DONE, "安全撤离完成")

    def runFunc(self):
        self.simulation_step += 1

        if self.state == PolishingState.SETTLE:
            self._hold_initial_and_calibrate()
            return

        self.measured_wrench = self._read_external_wrench_world()
        # 工件对工具的接触力与“压入”方向相反，因此取负投影。
        self.measured_normal_force = max(
            0.0,
            -float(np.dot(self.measured_wrench[:3], self.normal_axis_world)),
        )
        # print("self.measured_normal_force = ", self.measured_normal_force)

        if self.measured_normal_force > self.max_safe_force and self.state not in (
            PolishingState.RETRACT,
            PolishingState.DONE,
            PolishingState.FAULT,
        ):
            self.retract_start_pose = self.desired_pose.copy()
            self.reference_normal_force = 0.0
            self._set_state(
                PolishingState.RETRACT,
                f"力超限 {self.measured_normal_force:.2f} N，紧急撤离",
            )

        if self.state == PolishingState.APPROACH:
            self._run_approach()
        elif self.state == PolishingState.FORCE_RAMP:
            self._run_force_ramp()
        elif self.state == PolishingState.POLISH:
            self._run_polish()
        elif self.state == PolishingState.RETRACT:
            self._run_retract()
        elif self.state in (PolishingState.DONE, PolishingState.FAULT):
            return

        if not self._command_pose(self.desired_pose):
            self._set_state(PolishingState.FAULT, "IK求解失败")
            return

        if self.simulation_step % 200 == 0:
            print(
                f"[{self.data.time:7.3f}s] {self.state.name:10s} "
                f"F_ref={self.reference_normal_force:5.2f} N  "
                f"F_meas={self.measured_normal_force:5.2f} N  "
                f"dn={self.normal_offset:+.4f} m  "
                f"path=({self.path_offset[0]:+.4f},"
                f"{self.path_offset[1]:+.4f}) m"
            )


def run_headless(env, max_sim_time):
    env.runBefore()
    max_steps = int(max_sim_time / env.model.opt.timestep)
    for _ in range(max_steps):
        mujoco.mj_forward(env.model, env.data)
        env.runFunc()
        mujoco.mj_step(env.model, env.data)
        if env.state in (PolishingState.DONE, PolishingState.FAULT):
            break

    print("\n===== 打磨仿真结果 =====")
    print("state:", env.state.name)
    print("simulation_time:", round(float(env.data.time), 3), "s")
    print("reference_force:", round(env.reference_normal_force, 3), "N")
    print("measured_force:", round(env.measured_normal_force, 3), "N")
    print("normal_offset:", round(env.normal_offset, 6), "m")
    print("ik_failed:", env.ik_failed)
    print("control_finite:", bool(np.all(np.isfinite(env.data.ctrl))))
    if env.polish_force_samples:
        force_samples = np.asarray(env.polish_force_samples)
        print("polish_force_mean:", round(float(np.mean(force_samples)), 3), "N")
        print(
            "polish_force_mae:",
            round(float(np.mean(np.abs(force_samples - env.target_force))), 3),
            "N",
        )
        print("polish_force_max:", round(float(np.max(force_samples)), 3), "N")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Panda机械臂切向轨迹+法向恒力导纳打磨仿真"
    )
    parser.add_argument(
        "--target-force", type=float, default=8.0, help="目标法向力，单位 N"
    )
    parser.add_argument(
        "--trajectory",
        choices=("raster", "line"),
        default="raster",
        help="打磨轨迹：raster为二维蛇形栅格，line为原一维往复线",
    )
    parser.add_argument(
        "--stroke", type=float, default=0.02, help="往复打磨单边行程，单位 m"
    )
    parser.add_argument(
        "--path-frequency", type=float, default=0.03, help="往复轨迹频率，单位 Hz"
    )
    parser.add_argument(
        "--raster-width", type=float, default=0.04, help="栅格轨迹宽度，单位 m"
    )
    parser.add_argument(
        "--raster-height", type=float, default=0.03, help="栅格轨迹高度，单位 m"
    )
    parser.add_argument(
        "--raster-rows", type=int, default=4, help="栅格轨迹行数"
    )
    parser.add_argument(
        "--path-speed", type=float, default=0.01, help="栅格轨迹平均速度，单位 m/s"
    )
    parser.add_argument(
        "--polish-duration", type=float, default=22.0, help="恒力打磨时间，单位 s"
    )
    parser.add_argument("--headless", action="store_true", help="无界面快速仿真")
    parser.add_argument(
        "--max-sim-time", type=float, default=35.0, help="无界面最长仿真时间，单位 s"
    )
    args = parser.parse_args()

    mujoco_root = Path(__file__).resolve().parents[2]
    model_dir = mujoco_root / "model/franka_emika_panda"
    scene_xml = str(model_dir / "scene_polishing.xml")
    arm_xml = str(model_dir / "panda_pos.xml")

    polishing_env = PandaPolishingEnv(
        scene_xml,
        arm_xml,
        target_force=args.target_force,
        trajectory=args.trajectory,
        stroke=args.stroke,
        path_frequency=args.path_frequency,
        raster_width=args.raster_width,
        raster_height=args.raster_height,
        raster_rows=args.raster_rows,
        path_speed=args.path_speed,
        polish_duration=args.polish_duration,
    )
    if args.headless:
        run_headless(polishing_env, args.max_sim_time)
    else:
        polishing_env.run_loop()
