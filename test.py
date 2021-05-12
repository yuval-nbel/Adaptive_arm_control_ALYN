from arm import Model, Simulation
import numpy as np

BASE_DIR = '/home/nbel/NBEL/alyn_project/Adaptive_arm_control/Adaptive_arm_control/'
  
model_name  = 'NBEL'
model = Model(BASE_DIR + 'arm_models/{}/{}.xml'.format(model_name, model_name))

init_angles = {0: -np.pi/2, 1:0, 2:np.pi/2, 3:0, 4:np.pi/2, 5:0}
init_quat = [-np.pi, 0, -np.pi/2]
default_quat = [3.082675779951924, -0.037349176615183055, -1.1118398002232517]

target      = [np.array([ 0.20 , 0.10,-0.10,  0, 0, 0]), 
               np.array([ -0.10 , 0.10, -0.10 , 0, 0, 0]), 
               np.array([ 0.20 , 0.10,-0.10 ,  0, -0.3, 0]), 
               np.array([ -0.10 , 0.10, -0.10 ,  0, -0.3, 0])
]

simulation_ext = Simulation(model, init_angles, external_force=0,
                                  target=target, adapt=False)

simulation_ext.simulate()

'''
simulation_ext_adapt = Simulation(model, init_angles, external_force=1.5,
                                  target=target, adapt=True)                                  
simulation_ext_adapt.simulate()
'''