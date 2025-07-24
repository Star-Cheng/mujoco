import mujoco
import numpy as np
import glfw
import time
import math

def scroll_callback(window, xoffset, yoffset):
    global cam
    cam.distance *= 1 - 0.1 * yoffset

def limit_angle(angle):
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle

def interpolate_joints(start, target, alpha):
    """在两个关节配置之间进行插值"""
    return start + alpha * (target - start)

def calculate_joint_distance(q1, q2):
    """计算两个关节配置之间的欧氏距离"""
    return np.linalg.norm(q1 - q2)

def main(target_joints):
    global cam
    # 加载模型
    model = mujoco.MjModel.from_xml_path('./mujoco/model/franka_emika_panda/scene.xml')
    data = mujoco.MjData(model)

    # 初始化 GLFW
    if not glfw.init():
        return

    window = glfw.create_window(1200, 900, 'Panda Arm Control', None, None)
    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)
    glfw.set_scroll_callback(window, scroll_callback)

    # 初始化渲染器
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultCamera(cam)
    mujoco.mjv_defaultOption(opt)
    pert = mujoco.MjvPerturb()  # 修正了这里的拼写错误
    con = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
    
    scene = mujoco.MjvScene(model, maxgeom=10000)

    # 找到末端执行器的 body id
    end_effector_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'hand')
    if end_effector_id == -1:
        print("Warning: Could not find the end effector with the given name.")
        glfw.terminate()
        return

    # 设置目标关节角度 (确保长度正确)
    if len(target_joints) != 7:
        print("Error: Target joints must have exactly 7 values")
        glfw.terminate()
        return
    
    target_joints = np.array(target_joints, dtype=np.float64)
    
    # 初始关节角度
    start_joints = data.qpos[:7].copy()
    current_joints = start_joints.copy()
    
    # 运动参数
    max_velocity = 0.2  # 弧度/秒
    movement_threshold = 0.001  # 弧度
    interpolation_factor = 0.05  # 插值因子
    
    # 获取当前终端位置
    mujoco.mj_forward(model, data)
    start_pos = data.body(end_effector_id).xpos.copy()
    target_pos = np.zeros(3)  # 这里会被更新

    print(f"Starting joint positions: {list(start_joints)}")
    print(f"Target joint positions: {list(target_joints)}")
    
    # 跟踪时间和状态
    last_time = time.time()
    start_time = last_time
    movement_complete = False
    
    while not glfw.window_should_close(window):
        if movement_complete:
            # 保持最后位置
            data.qpos[:7] = target_joints
        else:
            # 计算时间增量
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            elapsed = current_time - start_time
            
            # 计算当前目标位置
            mujoco.mj_forward(model, data)
            current_pos = data.body(end_effector_id).xpos.copy()
            
            # 计算距离和目标位置
            distance_to_target = calculate_joint_distance(current_joints, target_joints)
            
            # 使用加权插值平滑接近目标
            alpha = min(interpolation_factor * (1.0 + distance_to_target), 0.3)
            current_joints = interpolate_joints(current_joints, target_joints, alpha)
            
            # 限制关节角度
            for i in range(7):
                current_joints[i] = limit_angle(current_joints[i])
            
            # 设置关节目标位置
            data.qpos[:7] = current_joints
            
            # 检查是否完成移动
            if distance_to_target < movement_threshold and elapsed > 0.5:
                movement_complete = True
                print("\nMovement completed!")
                print(f"Final joint positions: {list(current_joints)}")
                
                # 获取最终末端位置
                mujoco.mj_forward(model, data)
                final_pos = data.body(end_effector_id).xpos
                print(f"End effector start position: {start_pos}")
                print(f"End effector final position: {final_pos}")
                displacement = np.linalg.norm(start_pos - final_pos)
                print(f"Total displacement: {displacement:.4f} meters")

        # 模拟一步
        mujoco.mj_step(model, data)
        
        # 更新渲染场景
        viewport = mujoco.MjrRect(0, 0, 1200, 900)
        mujoco.mjv_updateScene(model, data, opt, pert, cam, mujoco.mjtCatBit.mjCAT_ALL.value, scene)
        mujoco.mjr_render(viewport, scene, con)
        
        # 显示运动状态
        if not movement_complete:
            progress = min(1.0, 1.0 - (distance_to_target / calculate_joint_distance(start_joints, target_joints)))
            print(f"\rMovement progress: {progress * 100:.1f}%", end="", flush=True)
        
        # 交换前后缓冲区
        glfw.swap_buffers(window)
        glfw.poll_events()
    
    glfw.terminate()


if __name__ == "__main__":
    # 示例目标关节角度 (单位: 弧度)
    target_joints = [0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4]
    main(target_joints)