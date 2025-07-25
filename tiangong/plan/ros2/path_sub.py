from moveit_msgs.msg import DisplayTrajectory
from rclpy.node import Node
import rclpy


class DisplayTrajectorySubscriber(Node):
    def __init__(self):
        super().__init__("display_trajectory_subscriber")
        self.subscription = self.create_subscription(DisplayTrajectory, "/display_planned_path", self.callback, 10)

    def callback(self, msg: DisplayTrajectory):
        for trajectory in msg.trajectory:
            for point in trajectory.joint_trajectory.points:
                positions = point.positions
                velocities = point.velocities
                self.get_logger().info(f"Received joint positions: {positions}")
                # self.get_logger().info(f"Received joint velocities: {velocities}")


def main(args=None):
    rclpy.init(args=args)
    node = DisplayTrajectorySubscriber()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
