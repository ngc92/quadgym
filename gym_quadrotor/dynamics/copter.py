import numpy as np

from gym_quadrotor.dynamics.coordinates import Euler


class CopterParams(object):
    def __init__(self):
        self._thrustfactor = 1.0
        self._dragfactor = 1.0
        self._mass = 1.0
        # we assume a diagonal matrix
        self._rotational_drag = np.array([0.01, 0.01, 0.01])
        self._translational_drag = np.array([0.01, 0.01, 0.01])
        self._arm_length = 1.0
        self._rotor_inertia = 1.0
        # we assume a diagonal matrix
        self._inertia = np.array([1.0, 1.0, 1.0])
        self._gravity = np.array([0.0, 0.0, -9.81])
        self._motor_factor = 3.0

    @property
    def thrust_factor(self):
        return self._thrustfactor

    @property
    def drag_factor(self):
        return self._dragfactor

    @property
    def mass(self):
        return self._mass

    @property
    def rotational_drag(self):
        return self._rotational_drag

    @property
    def translational_drag(self):
        return self._translational_drag

    @property
    def arm_length(self):
        return self._arm_length

    @property
    def rotor_inertia(self):
        return self._rotor_inertia

    @property
    def frame_inertia(self):
        return self._inertia

    @property
    def gravity(self):
        return self._gravity

    @property
    def motor_factor(self):
        return self._motor_factor


class DynamicsState(object):
    def __init__(self):
        self._position = np.zeros(3)
        self._attitude = Euler(0.0, 0.0, 0.0)
        self._velocity = np.zeros(3)
        self._rotorspeeds = np.zeros(4)
        self._angular_velocity = np.zeros(3)

    @property
    def position(self):
        return self._position

    @property
    def attitude(self):
        return self._attitude

    @property
    def velocity(self):
        return self._velocity

    @property
    def rotor_speeds(self):
        return self._rotorspeeds

    @rotor_speeds.setter
    def rotor_speeds(self, value):
        self._rotorspeeds[:] = value

    @property
    def angular_velocity(self):
        return self._angular_velocity

    @property
    def net_rotor_speed(self):
        return self._rotorspeeds[0] - self._rotorspeeds[1] + self._rotorspeeds[2] - self._rotorspeeds[3]