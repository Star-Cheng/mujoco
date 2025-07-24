from scipy.spatial.transform import Rotation as R
import numpy as np


def rpy_to_quaternions(roll: float, pitch: float, yaw: float, mode="xyzw"):
    """将欧拉角(roll, pitch, yaw)转换为四元数。
    参数:
        roll: 绕X轴的旋转角度(以度为单位)
        pitch: 绕Y轴的旋转角度(以度为单位)
        yaw: 绕Z轴的旋转角度(以度为单位)
        mode: 返回四元数的顺序，默认为"xyzw"，可选"wxyz"
    返回:
        四元数, 格式为numpy数组。
    """
    # 生成旋转对象（以XYZ欧拉角顺序）
    r = R.from_euler("xyz", [np.deg2rad(roll), np.deg2rad(pitch), np.deg2rad(yaw)])
    # 以xyzw顺序获取四元数
    xyzw = r.as_quat()  # 返回顺序是xyzw
    # 转换为wxyz顺序
    wxyz = [xyzw[3], xyzw[0], xyzw[1], xyzw[2]]
    if mode == "wxyz":
        return np.array(wxyz)
    else:
        return xyzw


if __name__ == "__main__":
    # 示例参数(以度为单位)
    roll, pitch, yaw = 10.0, 15.0, 20.0
    xyzw = rpy_to_quaternions(roll, pitch, yaw, mode="xyzw")
    wxyz = rpy_to_quaternions(roll, pitch, yaw, mode="wxyz")

    print("xyzw形式的四元数:", xyzw)
    print("wxyz形式的四元数:", wxyz)
