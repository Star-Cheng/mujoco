import mujoco
import mujoco.viewer
import numpy as np
import time

# 1. 加载模型
model_path = "./mujoco/model/franka_emika_panda/scene.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# 2. 使用实际存在的关节名称
JOINT_NAMES = [
    "joint1", "joint2", "joint3", "joint4",
    "joint5", "joint6", "joint7"
]

# 获取关节 ID 列表
joint_ids = [model.joint(name).id for name in JOINT_NAMES]

# 调试：打印所有关节信息
print("模型中的所有关节：")
for i in range(model.njnt):
    jnt = model.joint(i)
    print(f" - 关节 {jnt.name} (类型: {jnt.type}, ID: {jnt.id})")

print(f"选中的控制关节ID: {joint_ids}")

# 3. 创建查看器
with mujoco.viewer.launch_passive(model, data) as viewer:
    # 4. 重置到初始位置
    mujoco.mj_resetData(model, data)
    
    # 设置控制器参数
    KP = 150.0  # 增加位置控制增益
    KD = 25.0   # 增加阻尼增益
    
    # 目标关节角度 (弧度制) - 初始为安全位置
    target_qpos = np.array([0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4])
    
    # 5. 主控制循环
    step = 0
    while viewer.is_running():
        step += 1
        
        # 获取当前关节位置和速度
        current_qpos = data.qpos[joint_ids]
        current_qvel = data.qvel[joint_ids]
        
        # 计算 PD 控制力矩
        torque = KP * (target_qpos - current_qpos) - KD * current_qvel
        
        # 应用控制力矩 - 仅应用于前7个关节
        for i, jnt_id in enumerate(joint_ids):
            if i < len(data.ctrl):
                data.ctrl[i] = torque[i]
        
        # 步进模拟
        mujoco.mj_step(model, data)
        
        # 更新查看器
        viewer.sync()
        
        # 每0.01秒更新一次 (实时模拟)
        time.sleep(0.01)
        
        # 每3秒改变目标位置
        if step % 300 == 0:
            # 随机目标位置但保持在安全范围内
            new_target = np.zeros(7)
            new_target[0] = np.random.uniform(-2.9, 2.9)
            new_target[1] = np.random.uniform(-1.8, 1.7)
            new_target[2] = np.random.uniform(-2.9, 2.9)
            new_target[3] = np.random.uniform(-3.1, -0.9)
            new_target[4] = np.random.uniform(-2.9, 2.9)
            new_target[5] = np.random.uniform(0.0, 3.0)
            new_target[6] = np.random.uniform(-2.9, 2.9)
            
            print(f"\n更新目标位置: {new_target}")
            target_qpos = new_target