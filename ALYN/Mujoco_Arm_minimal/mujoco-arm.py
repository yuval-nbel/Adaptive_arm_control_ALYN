#from tmp.arm import Model, Simulation
#import numpy as np
from IK import *
from RoboticArm import *
from model import MuJoCo_Model as Model
#from simulation import Simulation, Controller
from utilities import *

BASE_DIR = '/home/nbel/NBEL/alyn_project/Adaptive_arm_control_ALYN/'
  
model_name  = 'NBEL'
model = Model(BASE_DIR + 'arm_models/{}/{}.xml'.format(model_name, model_name))


ik_model = viper300()
Robot = return_Robot(openu=True, speed=2)

# Example of how to set position in three different ways:  
position = Robot['Real']['Home']
position = [-0.017453292519943295, 0.7853981633974483, -0.0, 3.141592653589793, 0.7853981633974483] 
position = robot_to_model_position({1: 180, 2: 225, 3: 135, 4: 180, 5: 182, 6: 180, 7: 135, 8: 180, 9: 240})


## EE position in the physical model's configuration space
p = [1, -1, -1, -1, -1] # z x x y x: accounting for direction of rotation
q_dic = {i: p[i]*v for i, v in enumerate (position)}
#print(q_dic)


model.goto_null_position()                                  # Goto reference position
model.send_target_angles(q_dic)                             # Manipulate model
c = model.get_ee_position()                                 # Current position
q = model.get_ee_quaternion() 
e = euler_from_quaternion(q)

print('angles:', model_to_robot_position(position))
print("updated_position: ", position)
print("c: ", c)
print("e: ", e)
print("q: ", q)
model.visualize()
