from sensor_msgs.msg import JointState
from rclpy.node import Node
import numpy as np
import rclpy


class JointStateSubscriber(Node):
    def __init__(self):
        super().__init__("joint_state_subscriber")
        self.subscription = self.create_subscription(JointState, "/joint_states", self.callback, 10)

    def callback(self, msg: JointState):
        self.positions = np.array(msg.position)
        self.get_logger().info(f"Received joint positions: {self.positions}")

def main(args=None):
    rclpy.init(args=args)
    node = JointStateSubscriber()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
