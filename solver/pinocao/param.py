import pinocchio as pin
import numpy as np

if __name__ == "__main__":
    urdf_path = "./description/Robot600_12/urdf/Robot600_12.urdf"
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()

    # 初始化存储列表
    masses = []
    com_positions = []

    # 遍历所有关节（跳过索引0的根关节）
    for joint_id in range(1, model.njoints):
        # 获取关节对应的惯性信息
        inertia = model.inertias[joint_id]

        # 提取质量（使用属性而不是方法）
        mass = inertia.mass
        masses.append(mass)

        # 提取质心位置（在关节坐标系中）
        com = inertia.lever
        com_positions.append(com)

    # 转换为NumPy数组
    masses = np.array(masses)
    com_positions = np.array(com_positions)

    print("连杆数量:", len(masses))
    print("连杆质量 (kg):", masses)
    print("质心位置 (m):", com_positions)
    for ls in com_positions:
        print(list(ls))
