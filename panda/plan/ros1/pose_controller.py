#!/usr/bin/env python

import rospy
import sys
import moveit_commander
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

class PandaPoseController:
    def __init__(self):
        # 初始化MoveIt Commander和ROS节点
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('panda_pose_controller', anonymous=True)
        
        # 创建MoveGroupCommander对象
        self.robot = moveit_commander.MoveGroupCommander("panda_arm")
        rospy.loginfo("MoveGroup commander initialized for 'panda_arm'")
        
        # 设置规划参数（可选）
        self.robot.set_planning_time(5.0)  # 规划时间限制
        self.robot.set_num_planning_attempts(5)  # 尝试次数
        self.robot.set_max_velocity_scaling_factor(0.5)  # 速度比例因子

    def set_end_effector_pose(self, x, y, z, qx=0.0, qy=0.0, qz=0.0, qw=1.0):
        """设置并移动机械臂到目标位姿"""
        # 创建目标位姿
        target_pose = PoseStamped()
        target_pose.header.frame_id = self.robot.get_planning_frame()
        target_pose.pose.position.x = x
        target_pose.pose.position.y = y
        target_pose.pose.position.z = z
        target_pose.pose.orientation.x = qx
        target_pose.pose.orientation.y = qy
        target_pose.pose.orientation.z = qz
        target_pose.pose.orientation.w = qw
        
        # 设置目标位姿
        self.robot.set_pose_target(target_pose)
        
        # 规划运动轨迹
        rospy.loginfo("Planning trajectory to target pose...")
        plan_result = self.robot.plan()
        
        # 检查规划结果 - 返回的是元组 (success, trajectory, planning_time, error_code)
        if plan_result[0]:
            plan = plan_result[1]  # 获取轨迹对象
            rospy.loginfo("Plan found with %d points in %.3f seconds", 
                          len(plan.joint_trajectory.points), plan_result[2])
            
            # 执行运动规划
            rospy.loginfo("Executing planned trajectory...")
            success = self.robot.execute(plan, wait=True)
            
            if success:
                rospy.loginfo("Motion completed successfully!")
            else:
                rospy.logwarn("Motion execution failed!")
        else:
            rospy.logwarn("No valid plan found. Error: %s", plan_result[3])

    def run(self):
        """运行控制循环"""
        # 设置目标位姿
        rospy.loginfo("Setting target end-effector pose...")
        x = 0.5
        y = 0.0
        z = 0.3
        self.set_end_effector_pose(x, y, z)
        
        # 等待执行完成
        rospy.loginfo("Motion execution finished.")
        
        # 清理资源
        moveit_commander.roscpp_shutdown()
        rospy.loginfo("Node shutdown complete.")

if __name__ == '__main__':
    try:
        controller = PandaPoseController()
        controller.run()
    except rospy.ROSInterruptException:
        pass