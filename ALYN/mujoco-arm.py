#from tmp.arm import Model, Simulation
import numpy as np
from IK import *
from RoboticArm import *
from model import MuJoCo_Model as Model
from simulation import Simulation, Controller

BASE_DIR = '/home/nbel/NBEL/alyn_project/Adaptive_arm_control_ALYN/'
  
model_name  = 'NBEL'
model = Model(BASE_DIR + 'arm_models/{}/{}.xml'.format(model_name, model_name))


ik_model = viper300()

position = Robot['Real']['Home']
position = robot_to_model_position(Robot['Real']['Home'])
position = robot_to_model_position({1: 24, 2: 195, 3: 195, 4: 145, 5: 145, 6: 19, 7:193, 8:180, 9:180})
'''
updated_position = position + np.dot(np.linalg.pinv(ik_model.calc_J_numeric(position)), [0, 0.01, 0])
for i in range(10):
    updated_position = updated_position + np.dot(np.linalg.pinv(ik_model.calc_J_numeric(position)), [0, 0.01, 0])
'''
updated_position = position
## EE position in the physical model's configuration space
p = [1, -1, -1, 1, -1] # z x x y x: accounting for direction of rotation
q_dic = {i: p[i]*v for i, v in enumerate (updated_position)}
print(q_dic)

theta_dic = {i: np.rad2deg(p[i]*v) for i, v in enumerate (updated_position)}
print(theta_dic)

model.goto_null_position()                                  # Goto reference position
model.send_target_angles(q_dic)     
#model.send_target_angles(position)                          # Manipulate model
c = model.get_ee_position()                                 # Current position
model.visualize()
print(c)