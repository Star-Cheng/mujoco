import mujoco
import numpy as np
import glfw
import time
import socket
import threading
import select

def scroll_callback(window, xoffset, yoffset):
    global cam
    cam.distance *= 1 - 0.1 * yoffset

def limit_angle(angle):
    """将角度限制在[-π, π]范围内"""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle

def interpolate_joints(current, target, alpha):
    """在两个关节配置之间进行平滑插值"""
    return current * (1 - alpha) + target * alpha

def handle_client_connection(client_socket, global_state):
    """处理来自客户端的TCP连接"""
    print(f"New connection from {client_socket.getpeername()}")
    
    while global_state['running']:
        try:
            # 使用非阻塞方式读取数据
            ready = select.select([client_socket], [], [], 0.1)
            if ready[0]:
                data = client_socket.recv(1024)
                if not data:
                    break
                
                # 解析接收到的关节角度数据
                decoded = data.decode().strip()
                if decoded.startswith("TARGET:"):
                    angles = list(map(float, decoded.split(":")[1].split()))
                    if len(angles) == 7:
                        with global_state['lock']:
                            global_state['target_joints'] = np.array(angles, dtype=np.float64)
                            print(f"Received new target: {angles}")
                        response = "ACK"
                        client_socket.sendall(response.encode())
                    else:
                        print(f"Invalid joint data: {decoded}")
                        client_socket.sendall("INVALID_DATA".encode())
        except Exception as e:
            print(f"Connection error: {e}")
            break
    
    client_socket.close()
    print(f"Connection closed")

def start_tcp_server(global_state):
    """启动TCP服务器接收控制指令"""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('0.0.0.0', 65432))
    server.listen(5)
    server.setblocking(False)
    print(f"Visualizer running. Listening for control commands on port 65432...")
    
    while global_state['running']:
        try:
            client, addr = server.accept()
            client_thread = threading.Thread(target=handle_client_connection, 
                                            args=(client, global_state))
            client_thread.daemon = True
            client_thread.start()
        except BlockingIOError:
            time.sleep(0.1)
        except Exception as e:
            print(f"Server error: {e}")
            break
    
    server.close()

def main():
    global cam
    
    # 共享状态变量
    global_state = {
        'target_joints': np.array([0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4]),
        'current_joints': None,
        'running': True,
        'lock': threading.Lock()
    }
    
    # 加载模型
    model = mujoco.MjModel.from_xml_path('./mujoco/model/franka_emika_panda/scene.xml')
    data = mujoco.MjData(model)
    
    # 启动TCP服务器线程
    server_thread = threading.Thread(target=start_tcp_server, args=(global_state,))
    server_thread.daemon = True
    server_thread.start()
    
    # 初始化 GLFW
    if not glfw.init():
        global_state['running'] = False
        return
    
    window = glfw.create_window(1200, 900, 'Panda Arm Visualizer', None, None)
    if not window:
        glfw.terminate()
        global_state['running'] = False
        return
    
    glfw.make_context_current(window)
    glfw.set_scroll_callback(window, scroll_callback)
    
    # 初始化渲染器
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultCamera(cam)
    mujoco.mjv_defaultOption(opt)
    pert = mujoco.MjvPerturb()
    con = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
    scene = mujoco.MjvScene(model, maxgeom=10000)
    
    # 找到末端执行器的body id
    end_effector_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'hand')
    if end_effector_id == -1:
        print("Warning: Could not find the end effector with the given name.")
        global_state['running'] = False
    
    # 初始化关节角度
    start_joints = data.qpos[:7].copy()
    with global_state['lock']:
        global_state['current_joints'] = start_joints.copy()
    
    # 运动参数
    interpolation_factor = 0.1  # 插值平滑度因子
    movement_threshold = 0.001  # 运动完成阈值
    
    print("Panda Arm Visualizer running. Use a controller to send joint angles.")
    
    last_print_time = time.time()
    
    # 主渲染循环
    while not glfw.window_should_close(window) and global_state['running']:
        # 获取当前时间和关节状态
        with global_state['lock']:
            target_joints = global_state['target_joints']
            current_joints = global_state['current_joints']
        
        # 计算距离
        distance_to_target = np.linalg.norm(current_joints - target_joints)
        
        # 使用插值平滑接近目标
        if distance_to_target > movement_threshold:
            alpha = min(interpolation_factor * (1.0 + distance_to_target), 0.3)
            new_joints = interpolate_joints(current_joints, target_joints, alpha)
            
            # 限制关节角度范围
            for i in range(7):
                new_joints[i] = limit_angle(new_joints[i])
            
            # 更新当前关节状态
            with global_state['lock']:
                global_state['current_joints'] = new_joints
        
        # 更新模型关节位置
        data.qpos[:7] = global_state['current_joints']
        mujoco.mj_forward(model, data)
        
        # 更新渲染场景
        viewport = mujoco.MjrRect(0, 0, 1200, 900)
        mujoco.mjv_updateScene(model, data, opt, pert, cam, mujoco.mjtCatBit.mjCAT_ALL.value, scene)
        mujoco.mjr_render(viewport, scene, con)
        
        # 显示状态信息（每秒更新一次）
        current_time = time.time()
        if current_time - last_print_time > 1.0:
            end_effector_pos = data.body(end_effector_id).xpos
            print(f"Current joints: {global_state['current_joints']}")
            print(f"End effector position: {end_effector_pos}")
            print("-" * 50)
            last_print_time = current_time
        
        # 交换前后缓冲区
        glfw.swap_buffers(window)
        glfw.poll_events()
    
    # 清理资源
    global_state['running'] = False
    glfw.terminate()
    print("Visualizer shut down successfully.")

if __name__ == "__main__":
    main()