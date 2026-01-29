"""
Panda机械臂导纳控制仿真程序

本程序实现了基于MuJoCo物理引擎的Panda机械臂导纳控制（Admittance Control）。
导纳控制是一种力/位置混合控制方法，当机械臂受到外力时，会根据导纳模型
调整期望位置，使机械臂能够顺应外力运动。

导纳模型方程：M_d·ddq + B_d·dq + K_d·(q_des - q) = F_e
其中：
    - M_d: 虚拟质量矩阵（决定惯性响应）
    - B_d: 虚拟阻尼矩阵（决定阻尼特性）
    - K_d: 虚拟刚度矩阵（决定弹性响应）
    - F_e: 外力误差（测量力 - 参考力）
    - q_des: 期望位置
    - q: 当前位置
    - dq: 速度
    - ddq: 加速度
"""

import mujoco_viewer as mujoco_viewer
import numpy as np
from scipy.spatial.transform import Rotation as R
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

    def runBefore(self):
        """
        仿真循环开始前的初始化函数
        在run_loop()中会在主循环之前调用一次
        """
        # 获取初始关节位置（从keyframe中读取）
        self.initial_pos = self.model.key_qpos[0]
        # 设置前7个关节的控制量为初始位置（Panda有7个关节）
        self.data.ctrl[:7] = self.initial_pos[:7]

        # 末端执行器（End-Effector）的body名称
        self.ee_body_name = "ee_center_body"
        # 初始化逆运动学求解器（使用Pinocchio库）
        self.arm = PinocchioSolver("./description/panda_description/urdf/panda.urdf")

        # 保存上一次的关节角度（用于逆运动学求解的初始猜测）
        self.last_dof = self.data.qpos
        # 设置仿真时间步长为1ms（0.001秒）
        self.setTimestep(0.001)
        # 末端执行器期望速度的增量（6维：x, y, z, roll, pitch, yaw）
        self.delta_d_ee_des = np.zeros(6)
        # 末端执行器期望位置的增量（6维：x, y, z, roll, pitch, yaw）
        self.delta_ee_des = np.zeros(6)
        # 初始化计数器：前100步用于移动到初始位置
        self.first_goto_initial_pos_cnt = 100

        # 创建低通滤波器用于平滑末端执行器速度（6维，截止频率0.1Hz）
        self.vel_filter = lowpass_filter.LowPassOnlineFilter(6, 0.1, self.model.opt.timestep)
        # 创建低通滤波器用于平滑加速度（6维，截止频率0.1Hz）
        self.acc_filter = lowpass_filter.LowPassOnlineFilter(6, 0.1, self.model.opt.timestep)
        
        # 以下为注释掉的实时绘图代码（用于调试和可视化）
        # import src.matplot as mp
        # self.plot_manager = mp.MultiChartRealTimePlotManager()
        # self.plot_manager.addNewFigurePlotter("vel.x", "vel.x", row=0, col=0)
        # self.plot_manager.addNewFigurePlotter("delta.x", title="delta.x", row=1, col=0)
        # self.plot_manager.addNewFigurePlotter("delta.y", title="delta.y", row=2, col=0)
        # self.plot_manager.addNewFigurePlotter("delta.z", title="delta.z", row=3, col=0)
        
        # 逆运动学停止标志：当遇到边界或求解失败时停止更新
        self.ik_stop = False

    def runFunc(self):
        """
        每个仿真步骤执行的主控制函数
        实现导纳控制的核心逻辑：检测外力 -> 计算期望位置 -> 逆运动学求解 -> 更新关节角度
        """
        # ========== 第一阶段：初始化阶段（前100步） ==========
        if self.first_goto_initial_pos_cnt > 0:
            # 递减计数器
            self.first_goto_initial_pos_cnt -= 1
            # 将关节控制量设置为初始位置
            self.data.ctrl[:7] = self.initial_pos[:7]
            # 获取当前末端执行器位姿（6维：x, y, z, roll, pitch, yaw）
            self.ee_pos = self.getBodyPoseEulerByName(self.ee_body_name)
            # 初始化期望位置为当前位置
            self.desired_pos = self.ee_pos
            # 保存上一次的末端执行器位置（用于计算速度）
            self.last_ee_pos = self.ee_pos
            # 保存起始位置（用于计算位置增量）
            self.start_ee_pos = self.ee_pos
        else:
            # ========== 第二阶段：导纳控制阶段 ==========
            
            # --- 1. 获取当前末端执行器状态 ---
            # 获取当前末端执行器位姿
            self.now_ee_pos = self.getBodyPoseEulerByName(self.ee_body_name)
            # 通过数值微分计算当前速度：v = (p_now - p_last) / dt
            self.now_ee_vel = (self.now_ee_pos - self.last_ee_pos) / self.model.opt.timestep
            # 更新上一次位置
            self.last_ee_pos = self.now_ee_pos
            # 使用低通滤波器平滑速度信号（减少噪声）
            self.now_ee_vel_filter = self.vel_filter.update(self.now_ee_vel)
            # 计算位置误差：当前位置 - 期望位置
            ee_pos_err = self.now_ee_pos - self.desired_pos

            # --- 2. 力检测与计算 ---
            # 参考力（期望力，通常为零，表示不受力）
            F_ref = np.zeros(6)  # 6维：Fx, Fy, Fz, Mx, My, Mz
            # 测量力（实际检测到的力/力矩）
            # 注意：这里使用模拟力进行测试，实际应用中应从力传感器读取
            F_meas = np.array([0, 0, 0, 0, 0, 0])
            # 指定受力的轴索引（3表示z轴方向）
            self.axis_index = 3
            # 在z轴方向施加-5N的力（向下）
            F_meas[self.axis_index] = -5
            # 计算外力误差：测量力 - 参考力
            F_e = F_meas - F_ref

            # --- 3. 导纳控制参数设置 ---
            # 虚拟质量矩阵（对角矩阵，每个方向10kg，决定惯性响应）
            self.M_d = np.diag([10] * 6)
            # 虚拟阻尼矩阵（对角矩阵，每个方向1 N·s/m，决定阻尼特性）
            self.B_d = np.diag([1] * 6)
            # 虚拟刚度矩阵（对角矩阵，每个方向50 N/m，决定弹性响应）
            self.K_d = np.diag([50] * 6)
            
            # --- 4. 导纳控制方程求解 ---
            # 导纳模型：M_d·ddq + B_d·dq + K_d·(q_des - q) = F_e
            # 求解期望加速度：ddq = M_d^(-1) · (F_e - B_d·dq - K_d·(q_des - q))
            dd_ee = np.linalg.inv(self.M_d) @ (F_e - self.B_d @ self.now_ee_vel - self.K_d @ ee_pos_err)

            # --- 5. 积分计算期望速度和位置 ---
            # 通过积分加速度得到速度增量：dv = a * dt
            self.delta_d_ee_des += dd_ee * self.model.opt.timestep
            # 通过积分速度得到位置增量：dp = v * dt
            self.delta_ee_des += self.delta_d_ee_des * self.model.opt.timestep
            # 计算新的期望位置：起始位置 + 位置增量
            self.desired_pos[:6] = self.start_ee_pos[:6] + self.delta_ee_des[:6]

            # --- 6. 逆运动学求解 ---
            # 将期望位姿（位置+欧拉角）转换为齐次变换矩阵（4x4）
            tf = utils.transform2mat(
                self.desired_pos[0], self.desired_pos[1], self.desired_pos[2],  # x, y, z
                self.desired_pos[3], self.desired_pos[4], self.desired_pos[5]   # roll, pitch, yaw
            )
            # 使用逆运动学求解器计算关节角度
            # tf[:3, 3]: 位置向量（3x1）
            # tf[:3, :3]: 旋转矩阵（3x3）
            # self.last_dof[:9]: 上一次的关节角度（作为初始猜测，提高求解效率）
            flag, self.dof, _ = self.arm.ik(tf[:3, 3], tf[:3, :3], self.last_dof[:9])
            
            # --- 7. 边界检查和停止条件 ---
            # 如果z轴位置过小（接近0）或逆运动学求解失败，停止更新
            if self.desired_pos[self.axis_index] < 0.001 or not flag:
                self.ik_stop = True
            
            # --- 8. 更新关节角度 ---
            if not self.ik_stop:
                # 保存当前关节角度（用于下一次逆运动学求解的初始猜测）
                self.last_dof = self.dof
                # 更新前7个关节的位置（Panda机械臂有7个关节）
                self.data.qpos[:7] = self.dof[:7]
                
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
                self.data.qpos[:7] = self.last_dof[:7]


if __name__ == "__main__":
    """
    主函数：创建仿真环境并启动仿真循环
    
    使用说明：
    1. 加载场景和机械臂模型
    2. 创建PandaEnv环境实例
    3. 启动仿真循环（run_loop会自动调用runBefore和runFunc）
    """
    # 场景XML文件路径（包含环境、桌面、物体等）
    SCENE_XML = "./control/mujoco-learning/model/franka_emika_panda/scene_pos.xml"
    # 机械臂XML文件路径（包含Panda机械臂模型）
    ARM_XML = "./control/mujoco-learning/model/franka_emika_panda/panda_pos.xml"
    
    # 创建Panda机械臂导纳控制环境
    env = PandaEnv(SCENE_XML, ARM_XML)
    
    # 启动仿真循环
    # run_loop()会：
    #   1. 调用runBefore()进行初始化
    #   2. 循环执行：前向动力学 -> runFunc() -> 步进仿真 -> 同步显示
    env.run_loop()
