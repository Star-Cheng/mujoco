"""Panda机械臂键盘抓取放置（pick-and-place）MuJoCo仿真。

这个示例刻意保持简单：

1. 末端姿态始终保持垂直向下，键盘只控制末端XYZ位置。
2. 使用MuJoCo自身的末端雅可比做阻尼最小二乘逆运动学（DLS IK）。
3. 使用Panda原有的位置执行器跟踪IK得到的7个关节目标。
4. 方块进入绿色目标区、夹爪松开并稳定一段时间后，任务判定成功。

控制键从启动本程序的终端读取，不占用MuJoCo窗口的内置快捷键。
终端采用单字符即时输入，因此不需要按Enter。

键位：
    W / S：世界坐标系 +X / -X
    A / D：世界坐标系 +Y / -Y
    Q / E：世界坐标系 +Z / -Z
    O / C：打开 / 闭合夹爪
    R：重置机械臂和方块
    P：打印当前状态
    H：重新打印帮助
    X / Esc：结束程序

后续制作LeRobot数据集时，可以直接复用 ``get_state_vector`` 和
``get_action_vector``，再补充相机图像与任务文本即可。
"""

from __future__ import annotations

import argparse
import os
import select
import sys
import termios
import time
import tty
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


class TerminalKeyReader:
    """在终端中非阻塞地读取单个按键，并在退出时恢复终端设置。

    默认终端会等待用户按Enter后才把整行交给程序。这里临时切换到
    cbreak模式，让W/A/S/D等按键能够立即生效。cbreak仍保留Ctrl-C
    中断行为；无论正常退出还是抛出异常，``__exit__``都会恢复终端。
    """

    def __init__(self) -> None:
        self.file_descriptor: int | None = None
        self.original_settings: list | None = None

    def __enter__(self) -> "TerminalKeyReader":
        if not sys.stdin.isatty():
            raise RuntimeError(
                "终端键盘控制要求stdin连接到TTY，请直接在终端中运行本程序"
            )

        self.file_descriptor = sys.stdin.fileno()
        self.original_settings = termios.tcgetattr(self.file_descriptor)
        tty.setcbreak(self.file_descriptor)
        return self

    def read_key(self) -> str | None:
        """若已有按键则返回一个字符；没有输入时立即返回None。"""
        if self.file_descriptor is None:
            raise RuntimeError("TerminalKeyReader必须在with语句中使用")

        readable, _, _ = select.select([self.file_descriptor], [], [], 0.0)
        if not readable:
            return None
        return os.read(self.file_descriptor, 1).decode("utf-8", errors="ignore")

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.file_descriptor is not None and self.original_settings is not None:
            termios.tcsetattr(
                self.file_descriptor, termios.TCSADRAIN, self.original_settings
            )


class PandaPickPlaceKeyboard:
    """一个最小的Panda键盘抓取放置环境。"""

    # Panda夹爪执行器使用0~255控制：0表示闭合，255表示完全打开。
    GRIPPER_CLOSED = 0.0
    GRIPPER_OPEN = 255.0

    # 这些尺寸必须与scene_pick_place.xml保持一致。
    TABLE_TOP_Z = 0.30
    CUBE_HALF_SIZE = 0.025
    TARGET_RADIUS = 0.055

    def __init__(
        self,
        scene_xml: Path,
        position_step: float = 0.01,
        randomize_cube: bool = False,
        seed: int = 0,
    ) -> None:
        self.scene_xml = Path(scene_xml)
        self.position_step = float(position_step)
        self.randomize_cube = bool(randomize_cube)
        self.rng = np.random.default_rng(seed)

        if not self.scene_xml.exists():
            raise FileNotFoundError(f"找不到MuJoCo场景文件：{self.scene_xml}")
        if not 0.001 <= self.position_step <= 0.03:
            raise ValueError("position_step应在0.001~0.03 m之间")

        self.model = mujoco.MjModel.from_xml_path(str(self.scene_xml))
        self.data = mujoco.MjData(self.model)
        # IK使用独立的MjData，避免迭代求解时直接修改正在运行的动力学状态。
        self.ik_data = mujoco.MjData(self.model)
        self.model.opt.timestep = 0.002

        self.arm_joint_ids = np.array(
            [self._require_id(mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}") for i in range(1, 8)]
        )
        self.arm_qpos_adr = self.model.jnt_qposadr[self.arm_joint_ids].copy()
        self.arm_dof_adr = self.model.jnt_dofadr[self.arm_joint_ids].copy()
        self.arm_lower = self.model.jnt_range[self.arm_joint_ids, 0].copy()
        self.arm_upper = self.model.jnt_range[self.arm_joint_ids, 1].copy()

        self.ee_body_id = self._require_id(
            mujoco.mjtObj.mjOBJ_BODY, "ee_center_body"
        )
        self.cube_body_id = self._require_id(
            mujoco.mjtObj.mjOBJ_BODY, "pick_cube"
        )
        self.target_site_id = self._require_id(
            mujoco.mjtObj.mjOBJ_SITE, "place_target_center"
        )
        cube_joint_id = self._require_id(
            mujoco.mjtObj.mjOBJ_JOINT, "pick_cube_freejoint"
        )
        self.cube_qpos_adr = int(self.model.jnt_qposadr[cube_joint_id])
        self.home_key_id = self._require_id(
            mujoco.mjtObj.mjOBJ_KEY, "pick_place_home"
        )

        if self.model.nu < 8:
            raise ValueError("Panda模型应包含7个机械臂执行器和1个夹爪执行器")

        # 提高位置内环刚度，使笛卡尔键盘目标能够稳定地被跟踪。
        joint_kp = 3000.0
        joint_kv = 150.0
        self.model.actuator_gainprm[:7, 0] = joint_kp
        self.model.actuator_biasprm[:7, 1] = -joint_kp
        self.model.actuator_biasprm[:7, 2] = -joint_kv

        # 限制键盘目标，防止用户把末端移出Panda的主要工作空间。
        self.workspace_lower = np.array([0.25, -0.32, 0.325])
        self.workspace_upper = np.array([0.75, 0.32, 0.75])

        # 终端输入只生成待处理命令；实际MjData修改集中在仿真步开始时完成。
        self.pending_position_delta = np.zeros(3)
        self.pending_gripper_command: float | None = None
        self.pending_reset = False
        self.pending_print = False
        self.exit_requested = False

        self.desired_position = np.zeros(3)
        self.desired_rotation = np.eye(3)
        self.joint_target = np.zeros(7)
        self.gripper_command = self.GRIPPER_OPEN
        self.success_hold_start: float | None = None
        self.success_announced = False

        self.reset()

    def _require_id(self, object_type: mujoco.mjtObj, name: str) -> int:
        """按名称取得MuJoCo对象ID；名称错误时立即给出明确异常。"""
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"MuJoCo模型中找不到对象：{name}")
        return int(object_id)

    def reset(self) -> None:
        """恢复机械臂初始位姿，并按需随机化方块位置。"""
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key_id)

        if self.randomize_cube:
            # 只在桌面左半区随机化，避免方块与绿色目标区重叠。
            self.data.qpos[self.cube_qpos_adr : self.cube_qpos_adr + 3] = [
                self.rng.uniform(0.43, 0.57),
                self.rng.uniform(-0.20, -0.08),
                self.TABLE_TOP_Z + self.CUBE_HALF_SIZE + 0.001,
            ]
            self.data.qpos[self.cube_qpos_adr + 3 : self.cube_qpos_adr + 7] = [
                1.0,
                0.0,
                0.0,
                0.0,
            ]

        # 机械臂位置执行器的控制量就是目标关节角；夹爪255为打开。
        self.data.ctrl[:7] = self.data.qpos[self.arm_qpos_adr]
        self.data.ctrl[7] = self.GRIPPER_OPEN
        self.gripper_command = self.GRIPPER_OPEN
        mujoco.mj_forward(self.model, self.data)

        self.joint_target = self.data.qpos[self.arm_qpos_adr].copy()
        self.desired_position = self.data.xpos[self.ee_body_id].copy()
        # 第一帧的向下姿态作为整个任务的固定末端姿态。
        self.desired_rotation = self.data.xmat[self.ee_body_id].reshape(3, 3).copy()
        self.success_hold_start = None
        self.success_announced = False

        self.pending_position_delta.fill(0.0)
        self.pending_gripper_command = None
        self.pending_reset = False
        self.pending_print = False

        cube_position = self.data.xpos[self.cube_body_id]
        print(
            "场景已重置：方块位置 =",
            np.round(cube_position, 3),
            "，末端位置 =",
            np.round(self.desired_position, 3),
        )

    @staticmethod
    def _rotation_error(current: np.ndarray, desired: np.ndarray) -> np.ndarray:
        """计算世界坐标系中的小角度旋转误差。"""
        # 三个对应坐标轴叉乘之和，是SO(3)姿态误差的常用局部近似。
        return 0.5 * sum(
            np.cross(current[:, axis], desired[:, axis]) for axis in range(3)
        )

    def solve_ik(self, target_position: np.ndarray) -> tuple[bool, np.ndarray, float]:
        """使用MuJoCo雅可比求固定姿态下的7关节逆运动学。

        返回值：
            success：是否达到误差阈值。
            q：求得的7维关节目标；失败时返回原目标。
            error_norm：最终六维位姿误差范数。
        """
        q = self.joint_target.copy()
        self.ik_data.qpos[:] = self.data.qpos
        jacobian_position = np.zeros((3, self.model.nv))
        jacobian_rotation = np.zeros((3, self.model.nv))
        damping = 1e-3
        error_norm = np.inf

        for _ in range(120):
            self.ik_data.qpos[self.arm_qpos_adr] = q
            mujoco.mj_forward(self.model, self.ik_data)

            current_position = self.ik_data.xpos[self.ee_body_id]
            current_rotation = self.ik_data.xmat[self.ee_body_id].reshape(3, 3)
            position_error = target_position - current_position
            rotation_error = self._rotation_error(
                current_rotation, self.desired_rotation
            )
            pose_error = np.concatenate((position_error, rotation_error))
            error_norm = float(np.linalg.norm(pose_error))

            if np.linalg.norm(position_error) < 4e-4 and np.linalg.norm(rotation_error) < 2e-3:
                return True, q, error_norm

            mujoco.mj_jacBody(
                self.model,
                self.ik_data,
                jacobian_position,
                jacobian_rotation,
                self.ee_body_id,
            )
            jacobian = np.vstack(
                (
                    jacobian_position[:, self.arm_dof_adr],
                    jacobian_rotation[:, self.arm_dof_adr],
                )
            )
            regularized = jacobian @ jacobian.T + damping**2 * np.eye(6)
            delta_q = jacobian.T @ np.linalg.solve(regularized, pose_error)

            # 限制单次迭代关节变化，避免在奇异位形附近出现过大跳变。
            delta_norm = np.linalg.norm(delta_q)
            if delta_norm > 0.20:
                delta_q *= 0.20 / delta_norm
            q = np.clip(q + 0.7 * delta_q, self.arm_lower, self.arm_upper)

        return False, self.joint_target.copy(), error_norm

    def _queue_position_delta(self, delta: np.ndarray) -> None:
        """由终端按键提交一个世界坐标系平移增量。"""
        self.pending_position_delta += np.asarray(delta, dtype=float)

    def _handle_terminal_key(self, key: str) -> None:
        """把终端读取到的单字符转换成机械臂控制命令。"""
        if key in ("\x1b", "x", "X"):
            self.exit_requested = True
            return

        key = key.upper()

        step = self.position_step
        position_commands = {
            "W": np.array([step, 0.0, 0.0]),
            "S": np.array([-step, 0.0, 0.0]),
            "A": np.array([0.0, step, 0.0]),
            "D": np.array([0.0, -step, 0.0]),
            "Q": np.array([0.0, 0.0, step]),
            "E": np.array([0.0, 0.0, -step]),
        }
        if key in position_commands:
            self._queue_position_delta(position_commands[key])
            return

        if key == "O":
            self.pending_gripper_command = self.GRIPPER_OPEN
        elif key == "C":
            self.pending_gripper_command = self.GRIPPER_CLOSED
        elif key == "R":
            self.pending_reset = True
        elif key == "P":
            self.pending_print = True
        elif key == "H":
            self.print_help()

    def _consume_commands(self) -> None:
        """消费终端命令并更新位置执行器目标。"""
        delta = self.pending_position_delta.copy()
        self.pending_position_delta.fill(0.0)
        gripper_command = self.pending_gripper_command
        self.pending_gripper_command = None
        reset_requested = self.pending_reset
        self.pending_reset = False
        print_requested = self.pending_print
        self.pending_print = False

        if reset_requested:
            self.reset()
            return

        if np.any(np.abs(delta) > 0.0):
            old_position = self.desired_position.copy()
            requested_position = np.clip(
                old_position + delta, self.workspace_lower, self.workspace_upper
            )
            success, q_solution, error = self.solve_ik(requested_position)
            if success:
                self.desired_position = requested_position
                self.joint_target = q_solution
                self.data.ctrl[:7] = q_solution
            else:
                print(
                    "IK未收敛，本次移动已忽略。目标 =",
                    np.round(requested_position, 3),
                    f"，误差 = {error:.5f}",
                )

        if gripper_command is not None:
            self.gripper_command = float(gripper_command)
            self.data.ctrl[7] = self.gripper_command

        if print_requested:
            self.print_state()

    def _update_success(self) -> None:
        """检测方块是否在目标区内稳定放置，并且夹爪已经松开。"""
        cube_position = self.data.xpos[self.cube_body_id]
        target_position = self.data.site_xpos[self.target_site_id]
        horizontal_error = float(
            np.linalg.norm(cube_position[:2] - target_position[:2])
        )
        cube_on_table = abs(
            cube_position[2] - (self.TABLE_TOP_Z + self.CUBE_HALF_SIZE)
        ) < 0.025
        cube_inside_target = horizontal_error < (
            self.TARGET_RADIUS - 0.5 * self.CUBE_HALF_SIZE
        )
        gripper_is_open = self.gripper_command > 0.8 * self.GRIPPER_OPEN

        if cube_on_table and cube_inside_target and gripper_is_open:
            if self.success_hold_start is None:
                self.success_hold_start = float(self.data.time)
            elif (
                not self.success_announced
                and self.data.time - self.success_hold_start >= 0.5
            ):
                self.success_announced = True
                print("\n任务成功：方块已经稳定放入绿色目标区。按R开始下一次。")
        else:
            self.success_hold_start = None

    def get_state_vector(self) -> np.ndarray:
        """返回适合后续数据集使用的8维状态：7关节角＋夹爪宽度。"""
        arm_position = self.data.qpos[self.arm_qpos_adr].copy()
        # 两个手指各移动0~0.04 m，因此总开口宽度为二者之和。
        gripper_width = float(self.data.qpos[7] + self.data.qpos[8])
        return np.concatenate((arm_position, [gripper_width])).astype(np.float32)

    def get_action_vector(self) -> np.ndarray:
        """返回8维控制目标：7关节目标＋归一化夹爪开度。"""
        normalized_gripper = self.gripper_command / self.GRIPPER_OPEN
        return np.concatenate((self.data.ctrl[:7], [normalized_gripper])).astype(
            np.float32
        )

    def print_state(self) -> None:
        """打印调试抓取所需的最小状态信息。"""
        print("末端目标位置:", np.round(self.desired_position, 3))
        print("末端实际位置:", np.round(self.data.xpos[self.ee_body_id], 3))
        print("方块位置:", np.round(self.data.xpos[self.cube_body_id], 3))
        print("夹爪开度:", round(float(self.data.qpos[7] + self.data.qpos[8]), 4), "m")

    def print_help(self) -> None:
        """打印键位，避免使用者必须返回查看源码。"""
        print(
            """
Panda pick-and-place 键位
  请先点击启动mjpython的终端，再按以下控制键；无需按Enter。
  W/S : 世界坐标 +X/-X       A/D : 世界坐标 +Y/-Y
  Q/E : 世界坐标 +Z/-Z       O/C : 打开/闭合夹爪
  R   : 重置                  P   : 打印当前状态
  H   : 打印本帮助            X/Esc : 结束程序
  MuJoCo窗口未注册控制回调，因此其内置快捷键保持原样。
建议顺序：移到方块上方 -> 下降 -> C夹紧 -> Q抬起 -> 移到绿色圆盘
          -> E下降 -> O松开 -> Q抬起。
""".strip()
        )

    def step(self) -> None:
        """推进一个物理步；无界面测试和交互运行共用同一逻辑。"""
        self._consume_commands()
        mujoco.mj_step(self.model, self.data)
        self._update_success()

    def run_headless(self, duration: float) -> None:
        """无界面运行，用于自动检查模型、控制量和物理状态是否正常。"""
        steps = max(1, int(duration / self.model.opt.timestep))
        for _ in range(steps):
            self.step()
        print("无界面运行完成，simulation_time =", round(float(self.data.time), 3), "s")
        self.print_state()

    def run(self) -> None:
        """启动MuJoCo窗口，并从终端非阻塞读取机械臂控制键。"""
        self.print_help()
        with TerminalKeyReader() as terminal, mujoco.viewer.launch_passive(
            self.model, self.data
        ) as viewer:
            viewer.cam.distance = 1.35
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -25
            viewer.cam.lookat[:] = [0.45, 0.0, 0.38]

            sync_counter = 0
            while viewer.is_running() and not self.exit_requested:
                step_start = time.perf_counter()

                # 一次仿真循环把当前已经到达终端缓冲区的字符全部取完。
                # 这样快速连续按键不会积压到之后的控制周期。
                while True:
                    key = terminal.read_key()
                    if key is None:
                        break
                    self._handle_terminal_key(key)

                self.step()

                # 物理频率为500 Hz，没有必要每个物理步都刷新窗口。
                sync_counter += 1
                if sync_counter >= 5:
                    viewer.sync()
                    sync_counter = 0

                remaining = self.model.opt.timestep - (
                    time.perf_counter() - step_start
                )
                if remaining > 0:
                    time.sleep(remaining)


def default_scene_path() -> Path:
    """根据当前脚本位置定位场景，使程序不依赖启动时的工作目录。"""
    mujoco_root = Path(__file__).resolve().parents[2]
    return mujoco_root / "model/franka_emika_panda/scene_pick_place.xml"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Panda机械臂键盘pick-and-place MuJoCo仿真"
    )
    parser.add_argument(
        "--scene", type=Path, default=default_scene_path(), help="MuJoCo场景XML"
    )
    parser.add_argument(
        "--position-step",
        type=float,
        default=0.01,
        help="每次键盘平移增量，单位m，默认0.01",
    )
    parser.add_argument(
        "--randomize-cube",
        action="store_true",
        help="每次重置时随机化方块初始位置",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="方块随机位置的随机种子"
    )
    parser.add_argument(
        "--headless", action="store_true", help="不打开窗口，只做模型冒烟测试"
    )
    parser.add_argument(
        "--duration", type=float, default=2.0, help="无界面测试时长，单位s"
    )
    args = parser.parse_args()

    simulation = PandaPickPlaceKeyboard(
        scene_xml=args.scene,
        position_step=args.position_step,
        randomize_cube=args.randomize_cube,
        seed=args.seed,
    )
    if args.headless:
        simulation.run_headless(args.duration)
    else:
        simulation.run()


if __name__ == "__main__":
    main()
