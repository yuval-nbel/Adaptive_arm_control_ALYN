"""
Written by Dr. Elishai Ezra Tsur 
@ The Neuro-Biomorphic Engineering Lab (NBEL-lab.com)
@ The Open University of Israel
@ 15.9.2020
"""

from enum import Enum
import sympy as sp
import numpy as np
from SNN_IK import *

class Optimizer(Enum):
    """ Designation of an optimization method for inverse kinematic
    
    We support two optimization methods for inverse kinematic:
    1. Standard resolved motion (STD): Based on Pseudo-inversed jacobian
    2. Dampened least squares method (DLS) or the Levenberg–Marquardt algorithm: 
        see https://en.wikipedia.org/wiki/Levenberg–Marquardt_algorithm for a detailed description
    """

    STD = 1 
    DLS = 2 
    SNN = 3

class viper300:
    """ Describe the Viperx200 6DOF robotic arm by Trossen Robotic
    
    The class provides the properties, transformation matrices and jacobian of the ViperX 300.
    The arm is described in: https://www.trossenrobotics.com/viperx-300-robot-arm-6dof.aspx
    """
    
    def __init__ (self):
        
        # Robots' joints
        self.n_joints = 5
        self.q0 = sp.Symbol('q0') 
        self.q1 = sp.Symbol('q1') 
        self.q2 = sp.Symbol('q2') 
        self.q3 = sp.Symbol('q3')
        self.q4 = sp.Symbol('q4')
        
        # length of the robots' links
        self.l1 = (127-9)      * 1e-3
        self.l2 = (427-127)    * 1e-3
        self.l3 = (60)         * 1e-3
        self.l4 = (253-60)     * 1e-3
        self.l5 = (359-253)    * 1e-3
        self.l6 = (567-359)    * 1e-3
               
        # Calculate the transformation matrix for base to EE in operational space
        self.T = self.calculate_Tx().subs([('l1', self.l1), 
                                           ('l2', self.l2), 
                                           ('l3', self.l3), 
                                           ('l4', self.l4), 
                                           ('l5', self.l5),
                                           ('l6', self.l6)])
        
        # Calculate the Jacobian matrix for the EE
        self.J = self.calculate_J().subs([('l1', self.l1), 
                                          ('l2', self.l2), 
                                          ('l3', self.l3), 
                                          ('l4', self.l4), 
                                          ('l5', self.l5), 
                                          ('l6', self.l6)])
    
    def calculate_Tx(self):
        """ Calculate the transformation matrix for base to EE in operational space """
        
        q0 = self.q0
        q1 = self.q1
        q2 = self.q2
        q3 = self.q3
        q4 = self.q4
        
        l1 = sp.Symbol('l1')
        l2 = sp.Symbol('l2')
        l3 = sp.Symbol('l3')
        l4 = sp.Symbol('l4')
        l5 = sp.Symbol('l5')
        l6 = sp.Symbol('l6')
      
        # Rotate around the z axis, connected the world's ref axis to the arm's rotating base  
        T01 = sp.Matrix([[sp.cos(q0),  -sp.sin(q0), 0, 0],
                         [sp.sin(q0),  sp.cos(q0),  0, 0],
                         [0,           0,           1, 0],
                         [0,           0,           0, 1]])
        
        # Rotate around the x axis
        T12 = sp.Matrix([[1, 0,          0,           0 ],
                         [0, sp.cos(q1), -sp.sin(q1), 0 ],
                         [0, sp.sin(q1), sp.cos(q1),  l1],
                         [0, 0.        , 0,           1 ]])
        
        # Rotate around the x axis
        T23 = sp.Matrix([[1, 0,          0,           0 ],
                         [0, sp.cos(q2), -sp.sin(q2), l3],
                         [0, sp.sin(q2), sp.cos(q2),  l2],
                         [0, 0.        , 0,           1 ]])
        
        # Rotate around the y axis
        T34 = sp.Matrix([[sp.cos(q3), 0, sp.sin(q3), 0 ],
                         [0,          1, 0,          l4],
                         [-sp.sin(q3),0, sp.cos(q3), 0 ],
                         [0,          0, 0,          1 ]])
        
        # Rotate around the x axis
        T45 = sp.Matrix([[1, 0,          0,           0 ],
                         [0, sp.cos(q4), -sp.sin(q4), l5],
                         [0, sp.sin(q4), sp.cos(q4),  0],
                         [0, 0.        , 0,           1 ]])

        # Translate to the end effector
        T56 = sp.Matrix([[1, 0, 0, 0],
                         [0, 1, 0, l6],
                         [0, 0, 1, 0 ],
                         [0, 0, 0, 1 ]])

        T = T01 * T12 * T23 * T34 * T45 * T56
        x = sp.Matrix([0, 0, 0, 1])
        Tx = T * x
        
        return Tx
    
    def calculate_J(self):
        """ Calculate the Jacobian matrix for the EE """
    
        q = [self.q0, self.q1, self.q2, self.q3, self.q4]
        J = sp.Matrix.ones(3, 5)
        for i in range(3):     # x, y, z
            for j in range(5): # Five joints
                # Differentiate and simplify
                J[i, j] = sp.simplify(self.T[i].diff(q[j]))
                
        return J
    
    def get_xyz_symbolic(self, q):
        """ Calculate EE location in operational space by solving the for Tx symbolically """
        
        return np.array(self.T.subs([('q0', q[0]), 
                                     ('q1', q[1]), 
                                     ('q2', q[2]), 
                                     ('q3', q[3]),
                                     ('q4', q[4])]), dtype='float')

    def calc_J_symbolic(self, q): 
        """ Calculate the jacobian symbolically """
        return np.array(self.J.subs([('q0', q[0]), 
                                     ('q1', q[1]), 
                                     ('q2', q[2]), 
                                     ('q3', q[3]),
                                     ('q4', q[4])]), dtype='float')
    
    def get_xyz_numeric(self, q):
        """ Calculate EE location in operational space by solving the for Tx numerically
        
        Equation was derived symbolically and was then written here manually.
        Nuerical evaluation works faster then symbolically. 
        """
        
        q0 = q[0]
        q1 = q[1]
        q2 = q[2]
        q3 = q[3]
        q4 = q[4]
        
        
        T0 = (0.208*((np.sin(q0)*np.sin(q1)*np.cos(q2) + 
                      np.sin(q0)*np.sin(q2)*np.cos(q1))*np.cos(q3) + 
                     np.sin(q3)*np.cos(q0))*np.sin(q4) + 
              0.208*(np.sin(q0)*np.sin(q1)*np.sin(q2) 
                     - np.sin(q0)*np.cos(q1)*np.cos(q2))*np.cos(q4) 
              + 0.299*np.sin(q0)*np.sin(q1)*np.sin(q2) + 0.3*np.sin(q0)*np.sin(q1) 
              - 0.299*np.sin(q0)*np.cos(q1)*np.cos(q2) - 0.06*np.sin(q0)*np.cos(q1))
        
        T1 = (0.208*((-np.sin(q1)*np.cos(q0)*np.cos(q2) 
                      - np.sin(q2)*np.cos(q0)*np.cos(q1))*np.cos(q3) 
                     + np.sin(q0)*np.sin(q3))*np.sin(q4) 
              + 0.208*(-np.sin(q1)*np.sin(q2)*np.cos(q0) 
                       + np.cos(q0)*np.cos(q1)*np.cos(q2))*np.cos(q4) 
              - 0.299*np.sin(q1)*np.sin(q2)*np.cos(q0) - 0.3*np.sin(q1)*np.cos(q0) 
              + 0.299*np.cos(q0)*np.cos(q1)*np.cos(q2) + 0.06*np.cos(q0)*np.cos(q1))
        
        T2 = (0.208*(-np.sin(q1)*np.sin(q2) + np.cos(q1)*np.cos(q2))*np.sin(q4)*np.cos(q3) 
              + 0.208*(np.sin(q1)*np.cos(q2) + np.sin(q2)*np.cos(q1))*np.cos(q4) 
              + 0.299*np.sin(q1)*np.cos(q2) + 0.06*np.sin(q1) + 0.299*np.sin(q2)*np.cos(q1) + 0.3*np.cos(q1) + 0.118)
        
        T3 = 1

        return np.array([[T0],
                         [T1],
                         [T2], 
                         [T3]], dtype='float')
    
    def _calc_J_numeric(self, q):
        """ Calculate the Jacobian for q numerically
         
         Equation was derived symbolically and was then written here manually.
         Nuerical evaluation works faster then symbolically. 
         """
        
        q0 = q[0]
        q1 = q[1]
        q2 = q[2]
        q3 = q[3]
        q4 = q[4]
        
        J0 = (-0.208*np.sin(q0)*np.sin(q3)*np.sin(q4) + 0.3*np.sin(q1)*np.cos(q0) 
              + 0.208*np.sin(q4)*np.sin(q1 + q2)*np.cos(q0)*np.cos(q3) 
              - 0.06*np.cos(q0)*np.cos(q1) - 0.208*np.cos(q0)*np.cos(q4)*np.cos(q1 + q2) 
              - 0.299*np.cos(q0)*np.cos(q1 + q2))

        J1 = ((0.06*np.sin(q1) + 0.208*np.sin(q4)*np.cos(q3)*np.cos(q1 + q2) 
               + 0.208*np.sin(q1 + q2)*np.cos(q4) + 0.299*np.sin(q1 + q2) + 0.3*np.cos(q1))*np.sin(q0))
        
        J2 = ((0.208*np.sin(q4)*np.cos(q3)*np.cos(q1 + q2) 
               + 0.208*np.sin(q1 + q2)*np.cos(q4) + 0.299*np.sin(q1 + q2))*np.sin(q0))
        
        J3 = -0.208*(np.sin(q0)*np.sin(q3)*np.sin(q1 + q2) - np.cos(q0)*np.cos(q3))*np.sin(q4)
        
        J4 = (0.208*(np.sin(q0)*np.sin(q1 + q2)*np.cos(q3) 
                     + np.sin(q3)*np.cos(q0))*np.cos(q4) + 0.208*np.sin(q0)*np.sin(q4)*np.cos(q1 + q2))
        
        J5 = (0.3*np.sin(q0)*np.sin(q1) + 0.208*np.sin(q0)*np.sin(q4)*np.sin(q1 + q2)*np.cos(q3) 
              - 0.06*np.sin(q0)*np.cos(q1) - 0.208*np.sin(q0)*np.cos(q4)*np.cos(q1 + q2) 
              - 0.299*np.sin(q0)*np.cos(q1 + q2) + 0.208*np.sin(q3)*np.sin(q4)*np.cos(q0))
        
        J6 = (-(0.06*np.sin(q1) + 0.208*np.sin(q4)*np.cos(q3)*np.cos(q1 + q2) 
                + 0.208*np.sin(q1 + q2)*np.cos(q4) + 0.299*np.sin(q1 + q2) + 0.3*np.cos(q1))*np.cos(q0))
        
        J7 = (-(0.208*np.sin(q4)*np.cos(q3)*np.cos(q1 + q2) 
                + 0.208*np.sin(q1 + q2)*np.cos(q4) + 0.299*np.sin(q1 + q2))*np.cos(q0))
        
        J8 = 0.208*(np.sin(q0)*np.cos(q3) + np.sin(q3)*np.sin(q1 + q2)*np.cos(q0))*np.sin(q4)
        
        J9 = (0.208*(np.sin(q0)*np.sin(q3) 
                     - np.sin(q1 + q2)*np.cos(q0)*np.cos(q3))*np.cos(q4) 
              - 0.208*np.sin(q4)*np.cos(q0)*np.cos(q1 + q2))
        
        J10 = 0
        
        J11 = (-0.3*np.sin(q1) - 0.208*np.sin(q4)*np.sin(q1 + q2)*np.cos(q3) 
               + 0.06*np.cos(q1) + 0.208*np.cos(q4)*np.cos(q1 + q2) + 0.299*np.cos(q1 + q2))
        
        J12 = -0.208*np.sin(q4)*np.sin(q1 + q2)*np.cos(q3) + 0.208*np.cos(q4)*np.cos(q1 + q2) + 0.299*np.cos(q1 + q2)
        
        J13 = -0.208*np.sin(q3)*np.sin(q4)*np.cos(q1 + q2)
        
        J14 = -0.208*np.sin(q4)*np.sin(q1 + q2) + 0.208*np.cos(q3)*np.cos(q4)*np.cos(q1 + q2)

        
        return np.array([[J0,  J1,  J2,  J3,  J4],
                         [J5,  J6,  J7,  J8,  J9],
                         [J10, J11, J12, J13, J14]], dtype='float')


    def calc_J_numeric(self, q):
        """ Calculate the Jacobian for q numerically
         
         Equation was derived symbolically and was then written here manually.
         Nuerical evaluation works faster then symbolically. 
         """
        
        q0 = q[0]
        q1 = q[1]
        q2 = q[2]
        q3 = q[3]
        q4 = q[4]
        
        # position
        J0 = (-0.208*np.sin(q0)*np.sin(q3)*np.sin(q4) + 0.3*np.sin(q1)*np.cos(q0) 
              + 0.208*np.sin(q4)*np.sin(q1 + q2)*np.cos(q0)*np.cos(q3) 
              - 0.06*np.cos(q0)*np.cos(q1) - 0.208*np.cos(q0)*np.cos(q4)*np.cos(q1 + q2) 
              - 0.299*np.cos(q0)*np.cos(q1 + q2))

        J1 = ((0.06*np.sin(q1) + 0.208*np.sin(q4)*np.cos(q3)*np.cos(q1 + q2) 
               + 0.208*np.sin(q1 + q2)*np.cos(q4) + 0.299*np.sin(q1 + q2) + 0.3*np.cos(q1))*np.sin(q0))
        
        J2 = ((0.208*np.sin(q4)*np.cos(q3)*np.cos(q1 + q2) 
               + 0.208*np.sin(q1 + q2)*np.cos(q4) + 0.299*np.sin(q1 + q2))*np.sin(q0))
        
        J3 = -0.208*(np.sin(q0)*np.sin(q3)*np.sin(q1 + q2) - np.cos(q0)*np.cos(q3))*np.sin(q4)
        
        J4 = (0.208*(np.sin(q0)*np.sin(q1 + q2)*np.cos(q3) 
                     + np.sin(q3)*np.cos(q0))*np.cos(q4) + 0.208*np.sin(q0)*np.sin(q4)*np.cos(q1 + q2))
        
        J5 = (0.3*np.sin(q0)*np.sin(q1) + 0.208*np.sin(q0)*np.sin(q4)*np.sin(q1 + q2)*np.cos(q3) 
              - 0.06*np.sin(q0)*np.cos(q1) - 0.208*np.sin(q0)*np.cos(q4)*np.cos(q1 + q2) 
              - 0.299*np.sin(q0)*np.cos(q1 + q2) + 0.208*np.sin(q3)*np.sin(q4)*np.cos(q0))
        
        J6 = (-(0.06*np.sin(q1) + 0.208*np.sin(q4)*np.cos(q3)*np.cos(q1 + q2) 
                + 0.208*np.sin(q1 + q2)*np.cos(q4) + 0.299*np.sin(q1 + q2) + 0.3*np.cos(q1))*np.cos(q0))
        
        J7 = (-(0.208*np.sin(q4)*np.cos(q3)*np.cos(q1 + q2) 
                + 0.208*np.sin(q1 + q2)*np.cos(q4) + 0.299*np.sin(q1 + q2))*np.cos(q0))
        
        J8 = 0.208*(np.sin(q0)*np.cos(q3) + np.sin(q3)*np.sin(q1 + q2)*np.cos(q0))*np.sin(q4)
        
        J9 = (0.208*(np.sin(q0)*np.sin(q3) 
                     - np.sin(q1 + q2)*np.cos(q0)*np.cos(q3))*np.cos(q4) 
              - 0.208*np.sin(q4)*np.cos(q0)*np.cos(q1 + q2))
        
        J10 = 0
        
        J11 = (-0.3*np.sin(q1) - 0.208*np.sin(q4)*np.sin(q1 + q2)*np.cos(q3) 
               + 0.06*np.cos(q1) + 0.208*np.cos(q4)*np.cos(q1 + q2) + 0.299*np.cos(q1 + q2))
        
        J12 = -0.208*np.sin(q4)*np.sin(q1 + q2)*np.cos(q3) + 0.208*np.cos(q4)*np.cos(q1 + q2) + 0.299*np.cos(q1 + q2)
        
        J13 = -0.208*np.sin(q3)*np.sin(q4)*np.cos(q1 + q2)
        
        J14 = -0.208*np.sin(q4)*np.sin(q1 + q2) + 0.208*np.cos(q3)*np.cos(q4)*np.cos(q1 + q2)

        # oriantetion
        J15=0
        J16=1
        J17=1
        J18=0
        J19=1
        J20=0
        J21=0
        J22=1
        J23=0
        J24=0
        J25=1
        J26=0
        J27=0
        J28=0
        J29=0

        
        return np.array([[J0,  J1,  J2,  J3,  J4],
                         [J5,  J6,  J7,  J8,  J9],
                         [J10, J11, J12, J13, J14],
                         [J15,  J16,  J17,  J18,  J19],
                         [J20,  J21,  J22,  J23,  J24],
                         [J25, J26, J27, J28, J29]], dtype='float')
        J19=1
        J20=0
        J21=0
        J22=1
        J23=0
        J24=0
        J25=1
        J26=0
        J27=0
        J28=0
        J29=0

        
        return np.array([[J0,  J1,  J2,  J3,  J4],
                         [J5,  J6,  J7,  J8,  J9],
                         [J10, J11, J12, J13, J14],
                         [J15,  J16,  J17,  J18,  J19],
                         [J20,  J21,  J22,  J23,  J24],
                         [J25, J26, J27, J28, J29]], dtype='float')
    
class widow200:
    """ Describe the WidowX 200 5DOF robotic arm by Trossen Robotic
    
    The class provides the properties, transformation matrices and jacobian of the WidowX 200.
    The arm is described in: https://www.trossenrobotics.com/widowx-200-robot-arm.aspx
    """
    
    def __init__ (self):
        
        self.n_joints = 4
        
        self.l1 = (110.25-9)      * 1e-3
        self.l2 = (310.81-110.25) * 1e-3
        self.l3 = (50-14.1-15)    * 1e-3
        self.l4 = (200)           * 1e-3
        self.l5 = (422.43-250)    * 1e-3
        
        self.q0 = sp.Symbol('q0') 
        self.q1 = sp.Symbol('q1') 
        self.q2 = sp.Symbol('q2') 
        self.q3 = sp.Symbol('q3')
        
        self.T = self.calculate_Tx().subs([('l1', self.l1), 
                                           ('l2', self.l2), 
                                           ('l3', self.l3), 
                                           ('l4', self.l4), 
                                           ('l5', self.l5)])

        self.J = self.calculate_J().subs([('l1', self.l1), 
                                          ('l2', self.l2), 
                                          ('l3', self.l3), 
                                          ('l4', self.l4), 
                                          ('l5', self.l5)])
    
    def calculate_Tx(self):
        
        q0 = self.q0
        q1 = self.q1
        q2 = self.q2
        q3 = self.q3
        
        l1 = sp.Symbol('l1')
        l2 = sp.Symbol('l2')
        l3 = sp.Symbol('l3')
        l4 = sp.Symbol('l4')
        l5 = sp.Symbol('l5')
        
        T01 = sp.Matrix([[sp.cos(q0),  0, sp.sin(q0), 0],
                         [0,           1, 0,          0],
                         [-sp.sin(q0), 0, sp.cos(q0), 0],
                         [0,           0, 0,          1]])

        T12 = sp.Matrix([[sp.cos(q1), -sp.sin(q1), 0, 0 ],
                         [sp.sin(q1),  sp.cos(q1), 0, l1],
                         [0,           0         , 1, 0 ],
                         [0,           0         , 0, 1 ]])

        T23 = sp.Matrix([[sp.cos(q2), -sp.sin(q2), 0, l3 ],
                         [sp.sin(q2),  sp.cos(q2), 0, l2],
                         [0,           0         , 1, 0 ],
                         [0,           0         , 0, 1 ]])

        T34 = sp.Matrix([[sp.cos(q3), -sp.sin(q3), 0, l4],
                         [sp.sin(q3),  sp.cos(q3), 0, 0 ],
                         [0,           0         , 1, 0 ],
                         [0,           0         , 0, 1 ]])

        T45 = sp.Matrix([[1, 0, 0, l5],
                         [0, 1, 0, 0 ],
                         [0, 0, 1, 0 ],
                         [0, 0, 0, 1 ]])

        T = T01 * T12 * T23 * T34 * T45
        x = sp.Matrix([0, 0, 0, 1])
        Tx = T * x
        
        return Tx
    
    def calculate_J(self):
    
        q = [self.q0, self.q1, self.q2, self.q3]
        J = sp.Matrix.ones(3, 4)
        for i in range(3):     # x, y, z
            for j in range(4): # Four joints
                J[i, j] = sp.simplify(self.T[i].diff(q[j]))
                
        return J
        
    def get_xyz_symbolic(self, q):
        
        return np.array(self.T.subs([('q0', q[0]), 
                                     ('q1', q[1]), 
                                     ('q2', q[2]), 
                                     ('q3', q[3])]), dtype='float')

    def calc_J_symbolic(self, q): 
        
        return np.array(J_eval.subs([('q0', q[0]), 
                                     ('q1', q[1]), 
                                     ('q2', q[2]), 
                                     ('q3', q[3])]), dtype='float')
    
    def get_xyz_numeric(self, q):

        c0 = np.cos(q[0])
        c1 = np.cos(q[1])
        c2 = np.cos(q[2])
        c3 = np.cos(q[3])
        s0 = np.sin(q[0])
        s1 = np.sin(q[1])
        s2 = np.sin(q[2])
        s3 = np.sin(q[3])

        return np.array([[0.17243*(-s1*s2*c0 + c0*c1*c2*c3  + 0.17243*(-s1*c0*c2 - s2*c0*c1))*s3 - 
                          0.2*s1*s2*c0 - 0.20056*s1*c0 + 0.2*c0*c1*c2 + 0.0209*c0*c1        ],
                         [0.17243*(-s1*s2    + c1*c2)*s3    + 0.17243*(s1*c2 + s2*c1)*c3         + 
                          0.2*s1*c2    + 0.0209*s1     + 0.2*s2*c1    + 0.20056*c1 + 0.10125],
                         [0.17243*(s0*s1*s2  - s0*c1*c2)*c3 + 0.17243*(s0*s1*c2 + s0*s2*c1)*s3   + 
                          0.2*s0*s1*s2 + 0.20056*s0*s1 - 0.2*s0*c1*c2 - 0.0209*s0*c1        ],
                         [1]], dtype='float')

    def calc_J_numeric(self, q):

        c0 = np.cos(q[0])
        c1 = np.cos(q[1])
        c2 = np.cos(q[2])
        c3 = np.cos(q[3])
        s0 = np.sin(q[0])
        s1 = np.sin(q[1])
        s2 = np.sin(q[2])
        s3 = np.sin(q[3])

        s12  = np.sin(q[1] + q[2])
        c12  = np.cos(q[1] + q[2])
        s123 = np.sin(q[1] + q[2] + q[3])
        c123 = np.cos(q[1] + q[2] + q[3])

        return np.array([[(0.20056*s1 - 0.0209*c1 - 0.2*c12    - 0.17243*c123)*s0, 
                          -(0.0209*s1 + 0.2*s12 + 0.17243*s123  + 0.20056*c1)  *c0,
                          -(0.2*s12 + 0.17243*s123)*c0, -0.17243*s123*c0],
                         [ 0, -0.20056*s1 + 0.0209*c1 + 0.2*c12 + 0.17243*c123,
                           0.2*c12 + 0.17243*c123,0.17243*c123],
                         [(0.20056*s1 - 0.0209*c1 - 0.2*c12 - 0.17243*c123)*c0,
                          (0.0209*s1 + 0.2*s12 + 0.17243*s123 + 0.20056*c1)*s0,
                          (0.2*s12 + 0.17243*s123)*s0, 0.17243*s0*s123]], dtype='float')

def goto_target (arm, target, target_ref = 'reference', optimizer = Optimizer.DLS, monitor = True):
    """ Giving arm object, a target and optimizer, provides the required set of control signals 
    
    Returns the optimizing trajectory, error trace and arm configurazion to achieve the target.
    Target is defined in relative to the EE null position
    """
    
    q = np.array([[0]*arm.n_joints], dtype='float').T # Zero configuration of the arm
    xyz_c = (arm.get_xyz_numeric(q))[:-1]             # Current operational position of the arm 
    
    if target_ref == 'reference':
        xyz_t = xyz_c + target                        # Target operational position
    else:
        xyz_t = target

    count = 0
    trajectory_x = []
    trajectory_q = []
    error_tract  = []

    np.set_printoptions(precision=3)
    
    if monitor:
        print('Arm at: {}, going to: {}'.format(xyz_c, xyz_t))
    
    if optimizer is Optimizer.SNN:
        
        controller = SNN_IK(arm.get_xyz_numeric, arm.calc_J_numeric, xyz_t.flatten())
        
        while count < 100:
            
            output, error_SNN = controller.generate(q)
            xyz_c = (arm.get_xyz_numeric(output))[:-1] 
            xyz_d = xyz_t - xyz_c                      # Get vector to target
            error = np.sqrt(np.sum(xyz_d**2))          # Distance to target
            
            trajectory_q.append(output)
            trajectory_x.append(xyz_c)
            error_tract.append(error)
            
            if monitor:
                print('Error SNN: {}'.format(error_SNN))
                print('{}: error: {}, q: {}, at: {}'.format(count,error, output.T, xyz_c.T))

            if error < 0.01:
                break
                
            count += 1
        
        q = output
    
    else:
        
        controller = optimizer
        
        while count < 100:

            trajectory_q.append(np.copy(q))
            xyz_c = (arm.get_xyz_numeric(q))[:-1]      # Get current EE position
            trajectory_x.append(xyz_c)                 # Store to track coordinate trajectory
            xyz_d = xyz_t - xyz_c                      # Get vector to target
            error = np.sqrt(np.sum(xyz_d**2))          # Distance to target
            error_tract.append(error)                  # Store distance to track error

            kp = 0.1                                   # Proportional gain term
            ux = xyz_d * kp                            # direction of movement
            J_x = arm.calc_J_numeric(q)                # Calculate the jacobian

            # Solve inverse kinematics according to the designated optimizer
            if optimizer is Optimizer.STD: # Standard resolved motion
                u = np.dot(np.linalg.pinv(J_x), ux)

            elif optimizer is Optimizer.DLS: # Dampened least squares method
                u = np.dot(J_x.T, np.linalg.solve(np.dot(J_x, J_x.T) + np.eye(3) * 0.001, ux))

            q += u 
            count += 1

            # Stop when within 1mm accurancy (arm mechanical accurancy limit)
            if  error < .001:
                break

    if monitor:
        print('Arm config: {}, with error: {}, achieved @ step: {}'.format(
            np.rad2deg(q.T).astype(int), error, count))
    
    return q, trajectory_x, trajectory_q, error_tract, xyz_t, controller
