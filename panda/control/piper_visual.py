import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import mujoco_viewer
import numpy as np
import threading
import time
import mujoco
from rclpy.executors import MultiThreadedExecutor


class RobotSimulator(Node):
    def __init__(self, model_path):
        super().__init__("robot_mujoco_simulator")

        # 初始化MuJoCo仿真
        self.simulator = RobotMujocoSimulator(model_path)

        # 订阅joint_states话题
        self.joint_state_sub = self.create_subscription(JointState, "joint_states", self.joint_state_callback, 10)

        # 存储接收到的关节状态
        self.joint_positions = {}
        self.joint_velocities = {}
        self.joint_efforts = {}

        # 互斥锁，用于线程安全
        self.lock = threading.Lock()

        # 启动仿真线程
        self.sim_thread = threading.Thread(target=self.run_simulation)
        self.sim_thread.start()

        self.get_logger().info("Robot MuJoCo ROS2仿真器已启动")

    def joint_state_callback(self, msg: JointState):
        """接收joint_states话题的回调函数"""
        with self.lock:
            # 将关节状态存储到字典中，以关节名为键
            for i, name in enumerate(msg.name):
                if i < len(msg.position):
                    self.joint_positions[name] = msg.position[i]
                if i < len(msg.velocity):
                    self.joint_velocities[name] = msg.velocity[i]
                if i < len(msg.effort):
                    self.joint_efforts[name] = msg.effort[i]

    def run_simulation(self):
        """运行MuJoCo仿真循环"""
        while self.simulator.is_running():
            # 更新关节状态
            with self.lock:
                self.simulator.update_joints(self.joint_positions)

            # 运行仿真步
            self.simulator.step()

    def destroy_node(self):
        """清理资源"""
        self.simulator.close()
        super().destroy_node()


class RobotMujocoSimulator(mujoco_viewer.CustomViewer):
    def __init__(self, model_path):
        super().__init__(model_path, distance=3, azimuth=180, elevation=-30)

        # 获取模型中的所有关节
        self.joint_names = []
        self.joint_ids = {}

        # 遍历所有关节并记录
        for i in range(self.model.njnt):
            joint_id = i
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if joint_name:
                self.joint_names.append(joint_name)
                self.joint_ids[joint_name] = joint_id

        print(f"找到 {len(self.joint_names)} 个关节: {self.joint_names}")

        # 获取关节的qpos地址
        self.joint_qpos_adrs = {}
        for i, name in enumerate(self.joint_names):
            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jnt_id >= 0:
                # 获取关节在qpos数组中的起始地址
                adr = self.model.jnt_qposadr[jnt_id]
                self.joint_qpos_adrs[name] = adr

        # 存储当前关节位置
        self.current_joint_positions = {}

    def update_joints(self, joint_positions):
        """更新关节位置"""
        for name, position in joint_positions.items():
            if name in self.joint_qpos_adrs:
                adr = self.joint_qpos_adrs[name]
                if adr < len(self.data.qpos):
                    self.data.qpos[adr] = position
                    self.current_joint_positions[name] = position

    def step(self):
        """执行仿真步"""
        if self.is_running():
            # 前向动力学计算
            mujoco.mj_forward(self.model, self.data)

            # 可选：在这里添加控制逻辑

            # 仿真步进
            mujoco.mj_step(self.model, self.data)

            # 同步可视化
            self.sync()

            # 保持实时仿真速度
            time.sleep(self.model.opt.timestep)

    def run_loop(self):
        """重写父类的run_loop方法, 使用ROS2控制"""
        # ROS2会控制仿真循环，所以这里不执行任何操作
        pass

    def runBefore(self):
        """初始化设置"""
        pass

    def runFunc(self):
        """每帧执行的函数"""
        pass

    def is_running(self):
        """检查仿真是否在运行"""
        return super().is_running()

    def close(self):
        """关闭仿真"""
        if hasattr(self, "handle"):
            self.handle.close()


def main(args=None):
    rclpy.init(args=args)

    # 模型路径
    model_path = "./description/songling_description/mujoco_model/scene.xml"

    # 创建节点
    node = RobotSimulator(model_path)

    try:
        # 使用多线程执行器，允许仿真和ROS2回调同时运行
        executor = MultiThreadedExecutor()
        executor.add_node(node)

        # 运行节点
        executor.spin()

    except KeyboardInterrupt:
        pass
    finally:
        # 清理资源
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
