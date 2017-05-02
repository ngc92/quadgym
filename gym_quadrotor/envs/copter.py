""" This file contains classes and functions that are used for the simulation of the quadrotor
    helicopter.
    TODO I plan to change this to a more sophisticated model once I find the time.
"""

import numpy as np

class CopterParams(object):
    def __init__(self):
        self.l = 0.31       # Arm length
        self.b = 5.324e-5   # Thrust coefficient
        self.d = 8.721e-7   # Drag coefficient
        self.m = 0.723      # Mass
        self.I = np.array([[8.678e-3,0,0],[0,8.678e-3,0],[0,0,3.217e-2]]) # Inertia
        self.J = 7.321e-5   # Rotor inertia


class CopterStatus(object):
    def __init__(self):
        self.position = np.array([0.0, 0, 0])
        self.velocity = np.array([0.0, 0, 0])
        self.attitude = np.array([0.0, 0, 0])
        self.angular_velocity = np.array([0.0, 0, 0])

    @property
    def altitude(self):
        return self.position[2]


def calc_acceleration(status, params, control):
    b = params.b
    I = params.I
    l = params.l
    m = params.m
    J = params.J
    d = params.d
    g = 9.81

    attitude = status.attitude
    avel     = status.angular_velocity
    roll     = attitude[0]
    pitch    = attitude[1]
    yaw      = attitude[2]

    droll    = avel[0]
    dpitch   = avel[1]
    dyaw     = avel[2]

    # damn, have to calculate this
    U1s = control[0] / b
    U2s = control[1] / b
    U3s = control[2] / b
    U4s = control[3] / d
    U13 = (U1s + U4s) / 2
    U24 = (U1s - U4s) / 2
    O1 = np.sqrt(abs(U13 + U3s)/2)
    O3 = np.sqrt(abs(U13 - U3s)/2)
    O2 = np.sqrt(abs(U24 - U2s)/2)
    O4 = np.sqrt(abs(U24 + U2s)/2)
    Or = -O1 + O2 - O3 + O4

    c0 =  (4*control[0] + 1.0) *  m*g
    a0  = c0 * ( np.cos(roll)*np.sin(pitch)*np.cos(yaw) + np.sin(roll)*np.sin(yaw) ) / m
    a1  = c0 * ( np.cos(roll)*np.sin(pitch)*np.sin(yaw) + np.sin(roll)*np.cos(yaw) ) / m
    a2  = c0 * ( np.cos(roll)*np.cos(pitch) ) / m - g

    
    aroll  = (dpitch * dyaw * (I[1, 1] - I[2, 2]) + dpitch * Or * J + control[1] * l) / I[0, 0]
    apitch = (droll  * dyaw * (I[2, 2] - I[0, 0]) + droll * Or * J  + control[2] * l) / I[1, 1]
    ayaw   = (droll  * dyaw * (I[0, 0] - I[1, 1]) + control[3] * l) / I[2, 2]

    return np.array([a0, a1, a2]), np.array([aroll, apitch, ayaw])

def simulate(status, params, control, dt):
    ap, aa  = calc_acceleration(status, params, control)
    status.position += status.velocity * dt + 0.5 * ap * dt * dt
    status.velocity += ap * dt

    status.attitude += status.angular_velocity * dt + 0.5 * aa * dt * dt
    status.angular_velocity += aa * dt
