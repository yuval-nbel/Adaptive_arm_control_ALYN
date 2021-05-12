# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=trailing-whitespace
# pylint: disable=bad-whitespace
# pylint: disable=bad-continuation
# pylint: disable=invalid-name

from PS4_controller import PS4Controller
from RoboticArm import *
from IK import viper300
import numpy as np
import pprint

class RobotState:
    
    def __init__(self, init_state):
        self.state_chair = init_state
        self.state_model = robot_to_model_position(init_state)
    
    def update_chair(self, new_state):
        self.state_chair = new_state
        self.state_model = robot_to_model_position(self.state_chair)

    def update_model(self, update):
        self.state_model += update
        self.state_chair = model_to_robot_position(self.state_model)
        
robot_config = Robot    # Viper300 configuration
ik_model = viper300()   # Viper300 IK model
velocity_delta = 0.1   # Gain factor for actuation
state = RobotState(Robot['Real']['Home']) 

arm = RoboticArm(robot_config, COM_ID = '/dev/ttyUSB0')
arm.go_home()

# state.state_chair = engines_position
# state.state_model = joint_position


print('Home position: {}, at: {}'.format(state.state_chair, state.state_model))

def get_xyz_numeric_3d(axis):
    return np.array([axis[0][0],axis[1][0],axis[2][0]])

def actuation_function(robot_state, axis_dict, buttons_dict, arm):

    position = state.state_model

    current = get_xyz_numeric_3d(ik_model.get_xyz_numeric(position))

    J_x = ik_model.calc_J_numeric(robot_state.state_model) # Calculate the jacobian

    if not any ([buttons_dict[s]['value'] for s in [0,1,2,3,12]]):
        return
    
    if buttons_dict[3]['value']:   # Triangle press
        target = current + [0.1,0,0]
        # maybe check the new x is < 0.6
        # ...
        '''
        ux = np.array([1, 0, 0]) * velocity_delta
        '''

    elif buttons_dict[0]['value']: # Cross press
        target = current + [-0.1,0,0]
        # maybe check the new x is > 0
        # ...
        '''
        ux = np.array([-1, 0, 0]) * velocity_delta
        '''

    elif buttons_dict[1]['value']: # Circle press
        target = current + [0,0,0.1]
        # maybe check the new y is < 0.7
        # ...
        '''
        ux = np.array([0, 0, 1]) * velocity_delta
        '''
    elif buttons_dict[2]['value']: # Rectangle press

        # maybe check the new y is > -0.6
        # ...
        '''
        ux = np.array([0, 0, -1]) * velocity_delta
        '''

    elif buttons_dict[12]['value']: # Down press -> Terminate
        arm.destruct()
        return


    direction = target - current
    updated_position = (np.dot(np.linalg.pinv(ik_model.calc_J_numeric(position)), direction)*0.1)
    robot_state.update_model(updated_position)
    arm_actuation = robot_state.state_chair
    pprint.pprint(arm_actuation)

    #u = np.dot(np.linalg.pinv(J_x), ux)
    #robot_state.update_model(u)
    #arm_actuation = robot_state.state_chair
    #pprint.pprint(arm_actuation)

    updated_current = get_xyz_numeric_3d(ik_model.get_xyz_numeric(robot_state.state_model))

    pprint.pprint('at position: {}'.format(updated_current))
    arm.set_position(arm_actuation)

PS4 = PS4Controller()
PS4.listen(state, arm, actuation_function = actuation_function)
#PS4.listen(state, None, None)