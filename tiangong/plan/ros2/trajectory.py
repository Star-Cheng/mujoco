from moveit_msgs.msg import DisplayTrajectory
from sensor_msgs.msg import JointState
from rclpy.node import Node
import numpy as np
import rclpy


class Trajectory(Node):
    def __init__(self):
        super().__init__("trajectory_subscriber")
        self.display_trajectory_subscription = self.create_subscription(DisplayTrajectory, "/display_planned_path", self.display_trajectory_callback, 10)
        self.joint_state_subscription = self.create_subscription(JointState, "/joint_states", self.joint_state_callback, 10)
        self.joint_positions = np.zeros(14)  # 假设有7个关节

    def display_trajectory_callback(self, msg: DisplayTrajectory):
        for trajectory in msg.trajectory:
            for point in trajectory.joint_trajectory.points:
                velocities = point.velocities
                self.get_logger().info(f"Received joint velocities: {velocities}")

    def joint_state_callback(self, msg: JointState):
        self.joint_positions = np.array(msg.position)
        self.get_logger().info(f"Received joint positions: {self.joint_positions}")


def main(args=None):
    rclpy.init(args=args)
    node = Trajectory()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
