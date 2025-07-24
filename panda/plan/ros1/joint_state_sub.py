#!/usr/bin/env python

import rospy
from sensor_msgs.msg import JointState
import mujoco
import numpy as np
import glfw

class JointStateSubscriber:
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node("joint_state_subscriber", anonymous=True)
        
        # 加载模型
        self.model = mujoco.MjModel.from_xml_path('./mujoco/model/franka_emika_panda/scene.xml')
        self.data = mujoco.MjData(self.model)
        self.positions = np.zeros(7)  # 初始化为零位置

        # 打印所有 body 信息
        rospy.loginfo("All bodies in the model:")
        for i in range(self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            rospy.loginfo(f"ID: {i}, Name: {body_name}")

        # 初始化 GLFW
        if not glfw.init():
            rospy.logerr("Failed to initialize GLFW")
            return

        self.window = glfw.create_window(1200, 900, 'Panda Arm Control', None, None)
        if not self.window:
            glfw.terminate()
            rospy.logerr("Failed to create GLFW window")
            return

        glfw.make_context_current(self.window)

        # 设置鼠标滚轮回调函数
        glfw.set_scroll_callback(self.window, self.scroll_callback)

        # 初始化渲染器
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        mujoco.mjv_defaultCamera(self.cam)
        mujoco.mjv_defaultOption(self.opt)
        self.pert = mujoco.MjvPerturb()
        self.con = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)

        # 找到末端执行器的 body id
        self.end_effector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'hand')
        rospy.loginfo(f"End effector ID: {self.end_effector_id}")
        if self.end_effector_id == -1:
            rospy.logwarn("Could not find the end effector with the given name")
            glfw.terminate()
            return

        # 初始关节角度
        self.initial_q = self.data.qpos[:7].copy()
        rospy.loginfo(f"Initial joint positions: {self.initial_q}")

        # 创建关节状态订阅器
        self.subscription = rospy.Subscriber(
            "/joint_states",
            JointState,
            self.callback,
            queue_size=10
        )
        
        # 设置更新频率
        self.rate = rospy.Rate(100)  # 100Hz
        
        # 设置ROS节点关闭时的回调
        rospy.on_shutdown(self.cleanup)

    def scroll_callback(self, window, xoffset, yoffset):
        # 调整相机的缩放比例
        self.cam.distance *= 1 - 0.1 * yoffset

    def limit_angle(self, angle):
        # 限制角度在[-π, π]范围内
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def render(self):
        """渲染Mujoco场景"""
        # 获取当前末端执行器位置
        mujoco.mj_forward(self.model, self.data)
        
        # 更新渲染场景
        viewport = mujoco.MjrRect(0, 0, 1200, 900)
        mujoco.mjv_updateScene(
            self.model, 
            self.data, 
            self.opt, 
            self.pert, 
            self.cam, 
            mujoco.mjtCatBit.mjCAT_ALL.value, 
            self.scene
        )
        mujoco.mjr_render(viewport, self.scene, self.con)
        
        # 交换前后缓冲区
        glfw.swap_buffers(self.window)
        
    def callback(self, msg):
        """关节状态回调函数"""
        # 过滤Panda机械臂的7个关节（名称以panda_joint开头）
        self.positions = np.zeros(7)
        found_joints = 0
        
        for i, name in enumerate(msg.name):
            if "panda_joint" in name:
                # 尝试解析关节编号
                try:
                    joint_idx = int(name.replace("panda_joint", "")) - 1
                    if 0 <= joint_idx < 7:
                        self.positions[joint_idx] = msg.position[i]
                        found_joints += 1
                except ValueError:
                    pass
        
        # 如果找到所有关节，记录末端位置
        if found_joints == 7:
            mujoco.mj_forward(self.model, self.data)
            ee_pos = self.data.body(self.end_effector_id).xpos
            rospy.logdebug_throttle(
                0.5, 
                f"EE Position: x={ee_pos[0]:.3f}, y={ee_pos[1]:.3f}, z={ee_pos[2]:.3f}"
            )

    def run(self):
        """主循环"""
        rospy.loginfo("Starting visualization loop...")
        
        # 设置初始视角
        self.cam.azimuth = 135
        self.cam.elevation = -30
        self.cam.distance = 1.5
        self.cam.lookat[0] = 0.0
        self.cam.lookat[1] = 0.0
        self.cam.lookat[2] = 0.5
        
        while not rospy.is_shutdown() and not glfw.window_should_close(self.window):
            try:
                # 设置关节目标位置
                self.data.qpos[:7] = self.positions
                
                # 模拟一步
                mujoco.mj_step(self.model, self.data)
                
                # 渲染场景
                self.render()
                
                # 处理事件
                glfw.poll_events()
                
                # 控制循环频率
                self.rate.sleep()
            
            except Exception as e:
                rospy.logerr(f"Error in main loop: {str(e)}")
                break
        
        rospy.loginfo("Exiting visualization loop")

    def cleanup(self):
        """清理资源"""
        rospy.loginfo("Cleaning up resources...")
        if hasattr(self, 'window'):
            glfw.destroy_window(self.window)
        glfw.terminate()
        rospy.loginfo("Resources cleaned up")

if __name__ == "__main__":
    try:
        subscriber = JointStateSubscriber()
        subscriber.run()
    except rospy.ROSInterruptException:
        pass