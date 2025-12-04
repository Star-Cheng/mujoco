import numpy as np
import pinocchio as pin

# 1. 加载机器人 URDF 模型
model_path = "./data/urdf/piper_no_gripper_description.urdf"
# model_path = "/home/gym/code/ros2/fishbot/src/description/songling_description/urdf/piper_no_gripper_description.urdf"
model = pin.buildModelFromUrdf(model_path)
data = model.createData()

# 2. 定义关节位置、速度、加速度（假设速度加速度为零）
q = np.array([-0.508798383541387, 1.4817670816506658, -1.016060877341019, -0.07110471372624898, 1.052974591020699, 2.8536656868882884])  # q1-q6 关节角度 (rad)
v = np.zeros(6)  # 关节角速度 (rad/s)
a = np.zeros(6)  # 关节角加速度 (rad/s²)
tau_d = np.zeros(6)  # 干扰力矩，默认为零

# 3. 计算动力学项
pin.computeAllTerms(model, data, q, v)

# 获取质量矩阵 M(q)
M = pin.crba(model, data, q)  # 关节空间惯量矩阵

# 获取科氏力和离心力项 C(q, v) * v
Cv = pin.nle(model, data, q, v)  # 包含科氏力、离心力和重力
# 注意：nle 返回的是 C(q,v)*v + G(q)，所以我们需要减去重力项

# 获取重力项 G(q)
G = pin.computeGeneralizedGravity(model, data, q)

# 计算纯科氏力和离心力项：C(q,v)*v = nle - G
Cv_pure = Cv - G

# 4. 根据动力学方程计算控制力矩 τ
# τ = M(q)*a + C(q,v)*v + G(q) + τ_d
tau = M @ a + Cv_pure + G + tau_d

# 5. 打印结果
print("关节角度 q:", q)
print("关节角速度 v:", v)
print("关节角加速度 a:", a)
print("\n质量矩阵 M(q):")
print(M)
print("\n重力项 G(q):", G)
print("科氏力和离心力项 C(q,v)*v:", Cv_pure)
print("干扰力矩 τ_d:", tau_d)
print("\n计算得到的控制力矩 τ:", tau)

# 6. 使用逆动力学验证（直接计算力矩）
tau_inverse = pin.rnea(model, data, q, v, a)  # 逆动力学计算
print("\n逆动力学验证 (RNEA):", tau_inverse)
print("两者是否接近:", np.allclose(tau, tau_inverse))

# 7. 如果需要单独计算科氏力矩阵 C(q, v)
# Pinocchio 提供了计算科氏力矩阵的函数
C = pin.computeCoriolisMatrix(model, data, q, v)
print("\n科氏力矩阵 C(q, v):")
print(C)
print("验证 C(q,v)*v:", C @ v)