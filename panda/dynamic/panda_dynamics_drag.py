import mujoco_viewer as mujoco_viewer
import time
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import pinocchio as pin

class PandaEnv(mujoco_viewer.CustomViewer):
    def __init__(self, scene_xml, arm_xml):
        super().__init__(scene_xml, 3, azimuth=-45, elevation=-30)
        self.scene_xml = scene_xml
        self.arm_xml = arm_xml
        self.num_joints = self.model.nq
        print("Num of joints:", self.num_joints)
        
        self.initial_pos = self.model.key_qpos[0]
        self.data.qpos[:self.num_joints] = self.initial_pos[:self.num_joints]
        self.data.qvel[:self.num_joints] = np.zeros(self.num_joints)

        self.step = 0
        self.step_list = []
        self.dynamics_tau_list = []
        self.damping_tau_list = []

    def runBefore(self):
        self.pin_model = pin.RobotWrapper.BuildFromMJCF(self.arm_xml).model
        self.pin_data = self.pin_model.createData()
        self.last_v = np.zeros(self.num_joints)

    def runFunc(self):
        q = self.data.qpos[:self.num_joints]
        v = self.data.qvel[:self.num_joints]
        a = (v - self.last_v) / self.model.opt.timestep
        self.last_v = v.copy()
        # a = np.zeros(self.num_joints)
        v = np.concatenate((v, np.zeros(2)))[:self.num_joints]
        q = np.concatenate((q, np.zeros(2)))[:self.num_joints]
        a = np.concatenate((a, np.zeros(2)))[:self.num_joints]
        dynamics_tau = pin.rnea(self.pin_model, self.pin_data, q, v, a)
        DAMPING = -100
        damping_tau = DAMPING * v
        tau = dynamics_tau + damping_tau
        
        self.data.ctrl[:7] = tau[:7]
        print(f"Total Torque: {np.round(tau[:self.num_joints], 2)}")
        print(f"damping_tau: {np.round(damping_tau[:self.num_joints], 2)}")

        self.step += 1
        self.step_list.append(self.step)
        self.dynamics_tau_list.append(dynamics_tau[:self.num_joints].copy())
        self.damping_tau_list.append(damping_tau[:self.num_joints].copy())


if __name__ == "__main__":
    SCENE_XML = "./control/mujoco/model/franka_emika_panda/scene_tau.xml"
    # SCENE_XML = r"./data/xml/piper/scene.xml"
    SCENE_XML = r"./data/xml/unitree/scene.xml"
    ARM_XML = "./control/mujoco/model/franka_emika_panda/panda_tau.xml"
    # ARM_XML = r"./data/xml/piper/piper.xml"
    ARM_XML = r"./data/xml/unitree/z1.xml"
    env = PandaEnv(SCENE_XML, ARM_XML)
    env.run_loop()