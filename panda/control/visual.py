import mujoco_viewer
import numpy as np
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time,math

class Test(mujoco_viewer.CustomViewer):
    def __init__(self, path):
        super().__init__(path, 3, azimuth=180, elevation=-30)
        self.path = path
    
    def runBefore(self):
        pass
       
    def runFunc(self):
        pass

if __name__ == "__main__":
    test = Test("./description/mujoco_menagerie/unitree_g1/scene.xml")
    # test = Test("/home/gym/code/ros2/fishbot/src/description/songling_description/mujoco_model/scene.xml")
    # test = Test("./description/mujoco_menagerie/unitree_go2/scene.xml")
    # test = Test("./description/mujoco_menagerie/unitree_z1/scene.xml")
    test.run_loop()

    