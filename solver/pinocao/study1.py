import pinocchio as pin
import numpy as np
import time


def compute_gravity_compensation(model_, data_, q_):
    pin.computeGeneralizedGravity(model_, data_, q_)
    return data_


# 使用数值方法进行逆运动学
def inverse_kinematics(model, data, target_pose, initial_guess=None, max_iter=100, tol=1e-6):
    if initial_guess is None:
        q = pin.neutral(model)
    else:
        q = initial_guess.copy()

    for i in range(max_iter):
        # 前向运动学
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)

        # 计算当前位姿
        current_pose = data.oMf[frame_id]

        # 计算误差
        error_pos = target_pose.translation - current_pose.translation
        error_rot = pin.log3(target_pose.rotation @ current_pose.rotation.T)
        error = np.concatenate([error_pos, error_rot])

        if np.linalg.norm(error) < tol:
            print(f"逆运动学收敛于第 {i} 次迭代")
            return q

        # 计算雅可比矩阵
        J = pin.computeFrameJacobian(model, data, q, frame_id)

        # 使用伪逆求解
        dq = np.linalg.pinv(J) @ error * 0.1
        q = pin.integrate(model, q, dq)

    print("逆运动学未收敛")
    return q


# 正动力学计算
def forward_dynamics(model, data, q, q_dot, tau):
    # 使用ABA算法（Articulated Body Algorithm）
    q_ddot = pin.aba(model, data, q, q_dot, tau)
    return q_ddot


# 或者使用基本方程
def forward_dynamics_basic(model, data, q, q_dot, tau):
    # 计算质量矩阵
    M = pin.crba(model, data, q)
    # 计算非线性效应（科里奥利+重力）
    nle = pin.nonLinearEffects(model, data, q, q_dot)
    # 计算加速度
    q_ddot = np.linalg.inv(M) @ (tau - nle)
    return q_ddot


if __name__ == "__main__":
    # 1 创建模型
    urdf_path = "/home/gym/code/ros2/fishbot/src/description/Robot600_12/urdf/Robot600_12.urdf"
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()
    q = np.zeros(model.nq)
    data = compute_gravity_compensation(model, data, q)
    print(f"自由度数量: {model.nq}")
    print(f"关节数量: {model.njoints}")
    print(f"连杆数量: {model.nframes}")
    print(f"关节信息: {list(model.names)}")
    # print(f"连杆信息: {list(model.frames)}")
    # 获取连杆信息
    for frame in model.frames:
        print(f"连杆: {frame.name}, 父关节: {frame.parentJoint}")

    # 2 正运动学
    # 设置关节位置（配置向量）
    # q = pin.neutral(model)  # 中性位置
    # 或者自定义位置
    q = np.zeros(model.nq)
    q[0] = 0.1  # 设置第一个关节位置
    # 前向运动学计算
    pin.forwardKinematics(model, data, q)
    # 获取特定连杆的位姿
    frame_id = model.getFrameId("joint6")  # 获取末端执行器帧ID
    pin.updateFramePlacements(model, data)
    endeffector_pose = data.oMf[frame_id]
    print(f"末端位置: {endeffector_pose.translation}")
    # print(f"末端旋转矩阵:\n{endeffector_pose.rotation}")
    print(f"末端四元数：{pin.Quaternion(endeffector_pose.rotation)}")
    # 获取所有关节的位置
    for i in range(1, model.njoints):
        joint_pose = data.oMi[i]
        print(f"关节 {i} 位置: {joint_pose.translation}")
        # print(f"关节 {i} 旋转矩阵:\n{joint_pose.rotation}")
        print(f"关节 {i} 四元数：{pin.Quaternion(joint_pose.rotation)}")

    # 3 逆运动学

    # 4 雅可比矩阵
    # 计算相对于世界坐标系的雅可比矩阵
    J = pin.computeFrameJacobian(model, data, q, frame_id)
    print(f"雅可比矩阵形状: {J.shape}")

    # 计算相对于局部坐标系的雅可比矩阵
    J_local = pin.computeFrameJacobian(model, data, q, frame_id, pin.LOCAL)

    # 计算关节空间雅可比矩阵
    J_joint = pin.computeJointJacobians(model, data, q)

    # 5 动力学计算
    # 5.1 计算质量矩阵
    M = pin.crba(model, data, q)
    print(f"质量矩阵形状: {M.shape}")
    # 5.2 验证质量矩阵的对称性和正定性
    print(f"对称性误差: {np.max(np.abs(M - M.T))}")
    eigenvalues = np.linalg.eigvals(M)
    print(f"特征值(全部应大于0): {eigenvalues}")
    # 5.3 计算重力项
    g = pin.computeGeneralizedGravity(model, data, q)
    print(f"重力项: {g}")
    # 5.4 设置不同的重力方向
    model.gravity.linear = np.array([0, 0, -9.81])  # 地球重力
    # 5.5 计算质心
    com = pin.centerOfMass(model, data, q)
    print(f"质心位置: {com}")
    # 5.6 计算质心雅可比
    J_com = pin.jacobianCenterOfMass(model, data, q)
    print(f"质心雅可比形状: {J_com.shape}")
    # 5.7 逆动力学
    # 设置关节速度
    v = np.zeros(model.nv)  # 零速度
    v[0] = 1.0  # 第一个关节有速度
    # 设置关节加速度
    a = np.zeros(model.nv)  # 零加速度
    a[0] = 0.5  # 第一个关节有加速度
    # 计算逆动力学(RNEA算法)
    tau = pin.rnea(model, data, q, v, a)
    print(f"所需关节力矩: {tau}")
    # 5.8 正动力学
    # 计算正动力学(ABA算法)
    ddq = pin.aba(model, data, q, v, tau)
    print(f"关节加速度: {ddq}")
