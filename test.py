from arm import Model, Simulation
import numpy as np

BASE_DIR = '/home/nbel/NBEL/alyn_project/Adaptive_arm_control_ALYN/'
  
model_name  = 'NBEL'
model = Model(BASE_DIR + 'arm_models/{}/{}.xml'.format(model_name, model_name))

init_angles = {0: -np.pi/2, 1:0, 2:np.pi/2, 3:0, 4:np.pi/2, 5:0}

init_quat = np.array([-np.pi, 0, -np.pi/2])
init_point = np.array([0, 0.6, 0.4])



drinking = [1.57,0,1.5]
full_drinking = [-1.57, 0, -0.08] # init_quat + drinking

shelf = [-0.1, 1.57, 0]
full_shelf = [0.1, 1.57, -1.57] # init_quat + shelf

floor = [0, 1.57, -1.57]
full_floor = [-3.14, 1.57, -3.14] # init_quat + floor

'''
the arm is in it's start position in (0, 0.6, 0.4)
the q's are [0, 0, 0, 0, 0]
the angles are {1: 90, 2: 180, 3: 180, 4: 180, 5: 180, 6: 0, 7: 180, 8: 180, 9: 180}
the orientation is (-3.140655647182936, -0.0003622603130055021, -1.5834533187393665) (we can say: -np.pi, 0, -np.pi/2)

target = [how_much_to_move_form_0, 
         how_much_to_move_form_0.6,
         how_much_to_move_form_0.4,
         how_much_to_move_form_-np.pi,
         how_much_to_move_form_0,
         how_much_to_move_form_-np.pi/2]

'''


# Drinking (-1.5722844957856814, 3.847829373290325e-05, -0.06990636199343864)

target      = [
np.array([ 0.27,   -0.3, 0,   1.57,0,1.5]),
np.array([ 0.27,   -0.5,    0,   1.57,0,1.5]),
np.array([ 0.27,   -0.3, 0,   1.57,0,1.5]),
np.array([ 0.27,   -0.5,    0,   1.57,0,1.5]),
np.array([ 0.27,   -0.3, 0,   1.57,0,1.5]),
np.array([ 0.27,   -0.5,    0,   1.57,0,1.5]),
np.array([ 0.17,   -0.3, 0,   1.57,0,1.5]),
np.array([ 0.27,   -0.3,    0,   1.57,0,1.5]),
np.array([ 0.17,   -0.3, 0,   1.57,0,1.5]),
np.array([ 0.27,   -0.3,    0,   1.57,0,1.5]),
'''

# Shelf (1.384638056630099, 1.5269769702538658, 0.12312384027075265)
target      = [
np.array([ 0,0,0.1, -0.1, 1.57, 0]),
np.array([ 0.2,0,0.1, -0.1, 1.57, 0]),
np.array([ 0,0,0.15, -0.1, 1.57, 0]),
np.array([ 0.15,0,0.15,  -0.1, 1.57,0]),
np.array([ 0,0,-0.1, 0, 1.57, 0]),
np.array([ 0.2,0,0.1,  -0.1, 1.57,0]),
np.array([ 0,   0, -0.1,    0, 1.57, 0]),
np.array([ 0  , 0, 0.1,  -0.1, 1.57, 0]),
np.array([ 0,   -0.1, 0,    0, 1.57, 0]),
np.array([ 0  , -0.1, 0,  -0.1, 1.57, 0]),
'''
'''
np.array([ 0,0,0, 3.141592333436609, 3.659212831993035e-06, -3.0543151713606886]),
np.array([ 0.2,0,0,  3.141592333436609, 3.659212831993035e-06, -3.0543151713606886]),
np.array([ 0,0,0, 3.141592333436609, 3.659212831993035e-06, -3.0543151713606886]),
np.array([ 0.2,0,0,  3.141592333436609, 3.659212831993035e-06, -3.0543151713606886]),
np.array([ 0,0,0, 3.141592333436609, 3.659212831993035e-06, -3.0543151713606886]),
np.array([ 0.2,0,0,  3.141592333436609, 3.659212831993035e-06, -3.0543151713606886]),
np.array([ 0,0,0, 3.141592333436609, 3.659212831993035e-06, -3.0543151713606886]),
np.array([ 0.2,0,0,  3.141592333436609, 3.659212831993035e-06, -3.0543151713606886]),
'''
#    np.array([ 0.20 , 0.10,-0.10,  3.082675779951924, -0.037349176615183055, -1.1118398002232517]), 
 #              np.array([ -0.10 , 0.10, -0.10 , 0, 0, 0]), 
 #              np.array([ 0.20 , 0.10,-0.10 ,  0, 1, 0]), 
 #              np.array([ -0.10 , 0.10, -0.10 ,  0, 1, 0])
]

simulation_ext = Simulation(model, init_angles, external_force=0,
                                  target=target, adapt=False)

simulation_ext.simulate()

'''
simulation_ext_adapt = Simulation(model, init_angles, external_force=1.5,
                                  target=target, adapt=True)                                  
simulation_ext_adapt.simulate()
'''