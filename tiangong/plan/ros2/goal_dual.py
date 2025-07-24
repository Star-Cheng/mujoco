#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
"""
@File    :   goal_dual.py
@Time    :   2025/07/24 15:32:41
@Author  :   StarCheng
@Version :   1.0
@Site    :   https://star-cheng.github.io/Blog/
"""
from rclpy.callback_groups import ReentrantCallbackGroup
from pymoveit2.robots import tiangong as robot
from pymoveit2 import MoveIt2, MoveIt2State
from threading import Thread
from rclpy.node import Node
import rclpy


class GoalDual(Node):
    def __init__(self, node_name="goal_dual"):
        super().__init__(node_name)
        # Create node for this example
        # Declare parameters for position and orientation
        self.declare_parameter("position", [0.28298403, 0.24302717, 0.06437022])
        self.declare_parameter("quat_xyzw", [0.03085568, -0.70615245, -0.03083112, 0.706715])
        self.declare_parameter("synchronous", True)
        # If non-positive, don't cancel. Only used if synchronous is False
        self.declare_parameter("cancel_after_secs", 0.0)
        # Planner ID
        self.declare_parameter("planner_id", "RRTConnectkConfigDefault")
        # Declare parameters for cartesian planning
        self.declare_parameter("cartesian", False)
        self.declare_parameter("cartesian_max_step", 0.0025)
        self.declare_parameter("cartesian_fraction_threshold", 0.0)
        self.declare_parameter("cartesian_jump_threshold", 0.0)
        self.declare_parameter("cartesian_avoid_collisions", False)

    def goal(self, position, quat_xyzw):
        callback_group = ReentrantCallbackGroup()

        moveit2 = MoveIt2(
            node=self,
            joint_names=robot.joint_names(),
            base_link_name=robot.base_link_name(),
            end_effector_name=robot.end_effector_name(),
            group_name=robot.MOVE_GROUP_ARM,
            callback_group=callback_group,
        )
        moveit2.planner_id = self.get_parameter("planner_id").get_parameter_value().string_value

        # Spin the node in background thread(s) and wait a bit for initialization
        executor = rclpy.executors.MultiThreadedExecutor(2)
        executor.add_node(self)
        executor_thread = Thread(target=executor.spin, daemon=True, args=())
        executor_thread.start()
        self.create_rate(1.0).sleep()

        # Scale down velocity and acceleration of joints (percentage of maximum)
        moveit2.max_velocity = 0.5
        moveit2.max_acceleration = 0.5

        # Get parameters
        synchronous = self.get_parameter("synchronous").get_parameter_value().bool_value
        cancel_after_secs = self.get_parameter("cancel_after_secs").get_parameter_value().double_value
        cartesian = self.get_parameter("cartesian").get_parameter_value().bool_value
        cartesian_max_step = self.get_parameter("cartesian_max_step").get_parameter_value().double_value
        cartesian_fraction_threshold = self.get_parameter("cartesian_fraction_threshold").get_parameter_value().double_value
        cartesian_jump_threshold = self.get_parameter("cartesian_jump_threshold").get_parameter_value().double_value
        cartesian_avoid_collisions = self.get_parameter("cartesian_avoid_collisions").get_parameter_value().bool_value

        # Set parameters for cartesian planning
        moveit2.cartesian_avoid_collisions = cartesian_avoid_collisions
        moveit2.cartesian_jump_threshold = cartesian_jump_threshold

        # Move to pose
        self.get_logger().info(f"Moving to {{position: {list(position)}, quat_xyzw: {list(quat_xyzw)}}}")
        moveit2.move_to_pose(
            position=position,
            quat_xyzw=quat_xyzw,
            cartesian=cartesian,
            cartesian_max_step=cartesian_max_step,
            cartesian_fraction_threshold=cartesian_fraction_threshold,
        )
        if synchronous:
            moveit2.wait_until_executed()
        else:
            print("Current State: " + str(moveit2.query_state()))
            rate = self.create_rate(10)
            while moveit2.query_state() != MoveIt2State.EXECUTING:
                rate.sleep()

            # Get the future
            print("Current State: " + str(moveit2.query_state()))
            future = moveit2.get_execution_future()

            # Cancel the goal
            if cancel_after_secs > 0.0:
                # Sleep for the specified time
                sleep_time = self.create_rate(cancel_after_secs)
                sleep_time.sleep()
                # Cancel the goal
                print("Cancelling goal")
                moveit2.cancel_execution()

            # Wait until the future is done
            while not future.done():
                rate.sleep()

            # Print the result
            print("Result status: " + str(future.result().status))
            print("Result error code: " + str(future.result().result.error_code))

        rclpy.shutdown()
        executor_thread.join()
        exit(0)


def main() -> None:
    rclpy.init()
    node = GoalDual()
    left_pose = [0.28298403, 0.24302717, 0.06437022]
    left_quat = [0.03085568, -0.70615245, -0.03083112, 0.706715]
    node.goal(left_pose, left_quat)
    rclpy.spin(node)


if __name__ == "__main__":
    main()
