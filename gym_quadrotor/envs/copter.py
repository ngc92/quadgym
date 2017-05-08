""" This file contains classes and functions that are used for the simulation of the quadrotor
    helicopter.
    TODO I plan to change this to a more sophisticated model once I find the time.
"""

import numpy as np
from .propeller import Propeller
from .geo import make_quaternion

class CopterStatus(object):
    def __init__(self):
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.attitude = np.zeros(3)
        self.angular_velocity = np.zeros(3)
        self.rotor_speeds = np.array([1, 1, -1, -1.0]) * 200.0

    @property
    def altitude(self):
        return self.position[2]

    @property
    def rotation_matrix(self):
        return make_quaternion(self.attitude[0], self.attitude[1], self.attitude[2]).rotation_matrix

class CopterSetup(object):
    def __init__(self):
        # rotor aerodynamcis coefficients
        self.a = 5.324e-5
        self.b = 8.721e-7

        # more parameters
        self.mu = np.array([1, 1, 1, 1]) * 1e-7
        self.lm = np.array([1, 1, 1, 1]) * 1e-7

        self.l = 0.31   # Arm length
        self.m = 0.723  # mass
        self.J = 7.321e-5   # Rotor inertia
        self.iI = np.linalg.inv([[8.678e-3,0,0],[0,8.678e-3,0],[0,0,3.217e-2]]) # inverse Inertia (assumed to be diagonal)

        cfg = {'a': self.a, 'b': self.b, 'lm': self.lm, 'mu': self.mu, 'axis': [0,0,-1]}
        P1 = Propeller(d=1, p = self.l*np.array([1,0,0]), **cfg)
        P2 = Propeller(d=1, p = -self.l*np.array([1,0,0]), **cfg)
        P3 = Propeller(d=-1, p = self.l*np.array([0,1,0]), **cfg)
        P4 = Propeller(d=-1, p = -self.l*np.array([0,1,0]), **cfg)

        self.propellers = [P1, P2, P3, P4]


def calc_forces(status, setup):
    moment = np.zeros(3)
    force  = np.zeros(3)
    rot_t  = []

    for p, w in zip(setup.propellers, status.rotor_speeds):
        f, m, ma = p.get_dynamics(w, status)
        force  += f 
        moment += m
        rot_t += [ma]

    force += setup.m * np.array([0.0,0.0, -9.81])

    return force, moment, rot_t

def calc_accelerations(setup, status, control):
    force, moment, ma = calc_forces(status, setup)

    lin_acc = force / setup.m
    ang_acc = np.dot(setup.iI, moment)
    rot_acc = np.array(ma)

    for i, w in enumerate(ma):
        # TODO scale control to torque
        rot_acc[i] = w + setup.propellers[i].direction * (control[i] * 0.0075 + 0.025)
    #print(lin_acc)
    #print(status.rotor_speeds)
    return lin_acc, ang_acc, rot_acc / setup.J

def simulate(status, params, control, dt):
    ap, aa, ar  = calc_accelerations(params, status, control)
    # position update
    status.position += status.velocity * dt + 0.5 * ap * dt * dt
    status.velocity += ap * dt

    # angle update
    status.attitude += status.angular_velocity * dt + 0.5 * aa * dt * dt
    status.angular_velocity += aa * dt

    # rotor update
    status.rotor_speeds += ar * dt
