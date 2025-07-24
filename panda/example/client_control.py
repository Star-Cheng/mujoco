import socket
import time
import numpy as np

def main():
    # 连接到可视化服务器
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    try:
        client_socket.connect(('localhost', 65432))
        print("Connected to visualizer successfully.")
    except ConnectionRefusedError:
        print("Could not connect to visualizer. Is it running?")
        return
    
    # 演示循环 - 发送多个关节角度
    try:
        # 示例目标位置序列
        targets = [
            [0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4],      # 初始位置
            [0.5, -np.pi/6, 0.2, -2.5, 0.1, np.pi/3, np.pi/3],            # 位置1
            [-0.3, -np.pi/3, -0.3, -2.0, -0.2, 2*np.pi/3, np.pi/6],       # 位置2
            [0.7, -np.pi/5, 0.4, -2.8, 0.3, np.pi/4, np.pi/2],            # 位置3
            [0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4]       # 返回初始位置
        ]
        
        for target in targets:
            # 格式化关节角度
            joint_str = " ".join(f"{angle:.4f}" for angle in target)
            message = f"TARGET: {joint_str}"
            
            # 发送指令
            client_socket.sendall(message.encode())
            print(f"Sent target: {target}")
            
            # 等待确认
            response = client_socket.recv(1024)
            print(f"Server response: {response.decode()}")
            
            # 等待2秒发送下一个指令
            time.sleep(2.0)
            
    except KeyboardInterrupt:
        print("\nControl interrupted by user.")
    finally:
        client_socket.close()
        print("Disconnected from visualizer.")

if __name__ == "__main__":
    main()