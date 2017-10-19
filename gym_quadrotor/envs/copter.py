""" This file contains classes and functions that are used for the simulation of the quadrotor
    helicopter.
    TODO I plan to change this to a more sophisticated model once I find the time.
"""
from collections import namedtuple

import numpy as np
from .propeller import Propeller
from .geo import make_quaternion


class CopterStatus(object):
    def __init__(self, pos=None, vel=None, att=None, avel=None, rspeed=None):
        self.position = np.zeros(3) if pos is None else pos
        self.velocity = np.zeros(3) if vel is None else vel
        self._attitude = np.zeros(3) if att is None else att
        self.angular_velocity = np.zeros(3) if avel is None else avel
        self.rotor_speeds = np.array([1, 1, -1, -1.0]) * 180.0 if rspeed is None else rspeed
        self._rotation_matrix = None

    @property
    def altitude(self):
        return self.position[2]

    @property
    def attitude(self):
        return self._attitude

    @attitude.setter
    def attitude(self, value):
        self._attitude = value
        self._rotation_matrix = None

    @property
    def rotation_matrix(self):
        if self._rotation_matrix is None:
            self._rotation_matrix = make_quaternion(self.attitude[0], self.attitude[1], self.attitude[2]).rotation_matrix
        return self._rotation_matrix

    def to_world_direction(self, axis):
        return np.dot(self.rotation_matrix, axis)

    def to_world_position(self, pos):
        return self.to_world_direction(pos) + self.position

    def to_local_direction(self, axis):
        return np.dot(np.linalg.inv(self.rotation_matrix), axis)

    def to_local_position(self, pos):
        return self.to_local_direction(pos - self.position)

    def __repr__(self):
        return "CopterStatus(%r, %r, %r, %r, %r)"%(self.position, self.velocity, self.attitude, self.angular_velocity, 
            self.rotor_speeds)

    def __str__(self):
        return "CopterStatus(pos=%s, vel=%s, att=%s, avel=%s, rspeed=%s)"%(self.position, self.velocity, self.attitude, self.angular_velocity, 
            self.rotor_speeds)


class CopterSetup(object):
    def __init__(self):
        # rotor aerodynamcis coefficients
        self.a = 5.324e-5
        self.b = 8.721e-7

        # more parameters
        self.mu = np.array([1, 1, 1, 1]) * 1e-7
        self.lm = np.array([1, 1, 1, 1]) * 1e-7

        # motor data
        self.motor_torque = 0.035  # [NM]

        self.l = 0.31   # Arm length
        self.m = 0.723  # mass
        self.J = 7.321e-5   # Rotor inertia
        self.iI = np.linalg.inv([[8.678e-3,0,0],[0,8.678e-3,0],[0,0,3.217e-2]]) # inverse Inertia (assumed to be diagonal)

        cfg = {'a': self.a, 'b': self.b, 'lm': self.lm, 'mu': self.mu, 'axis': [0,0,-1]}
        P1 = Propeller(d=1,  p = self.l*np.array([1, 0, 0.0]), **cfg)
        P2 = Propeller(d=1,  p = self.l*np.array([-1, 0, 0.0]), **cfg)
        P3 = Propeller(d=-1, p = self.l*np.array([0, 1, 0.0]), **cfg)
        P4 = Propeller(d=-1, p = self.l*np.array([0, -1, 0.0]), **cfg)

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

    rot_acc = np.array(ma)
    motor_torque = np.clip(control, 0.0, 1.0) * setup.motor_torque

    for i, w in enumerate(ma):
        torque      = setup.propellers[i].direction * motor_torque[i]
        rot_acc[i]  = (w + torque) / setup.J
        # the motor creates the reverse toruqe on the helicopter.
        moment     -= torque * status.to_world_direction(setup.propellers[i].axis)
    
    lin_acc = force / setup.m
    ang_acc = np.dot(setup.iI, moment)

    return lin_acc, ang_acc, rot_acc


def simulate(status, params, control, dt):
    ap, aa, ar  = calc_accelerations(params, status, control)
    # position update
    status.position += status.velocity * dt + 0.5 * ap * dt * dt
    status.velocity += ap * dt

    # angle update
    aa = status.to_local_direction(aa)
    status.attitude += status.angular_velocity * dt + 0.5 * aa * dt * dt
    status.angular_velocity += aa * dt

    # rotor update
    status.rotor_speeds += ar * dt


AccelerationData = namedtuple("AccelerationResult", ("linear", "angular", "rotor_speeds"))


def calculate_equilibrium_acceleration(setup, strength):
    control = np.ones(4) * strength
    status = CopterStatus()

    for i in range(50):
        #print(status)
        simulate(status, setup, control, 0.1)
        status.position = np.array([0.0, 0.0, 0.0])
        status.velocity = np.array([0.0, 0.0, 0.0])
        status.attitude = np.array([0.0, 0.0, 0.0])
        status.angular_velocity = np.array([0.0, 0.0, 0.0])

    result = calc_accelerations(setup, status, control)
    return AccelerationData(linear=result[0], angular=result[1], rotor_speeds=status.rotor_speeds)
