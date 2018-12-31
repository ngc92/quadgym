import numpy as np

from gym_quadrotor.dynamics.coordinates import Euler


class CopterParams(object):
    def __init__(self, thrust_factor=5.723e-6, drag_factor=1.717e-7, mass=0.723, rotational_drag=None,
                 translational_drag=None, arm_length=0.31, rotor_inertia=7.321e-5, frame_inertia=None,
                 gravity=None, max_rotor_speed=500.0, rotor_speed_half_time=1.0 / 16):

        if rotational_drag is None:
            rotational_drag = np.array([1, 1, 1]) * 1e-4

        if translational_drag is None:
            translational_drag = np.array([1, 1, 1]) * 1e-4

        if frame_inertia is None:
            frame_inertia = np.array([8.678, 8.678, 32.1]) * 1e-3

        if gravity is None:
            gravity = np.array([0.0, 0.0, -9.81])

        self._thrustfactor = thrust_factor
        self._dragfactor = drag_factor
        self._mass = mass
        # we assume a diagonal matrix
        self._rotational_drag = np.asarray(rotational_drag)
        self._translational_drag = np.asarray(translational_drag)
        self._arm_length = arm_length
        self._rotor_inertia = rotor_inertia
        # we assume a diagonal matrix
        self._inertia = np.asarray(frame_inertia)
        self._gravity = np.asarray(gravity)
        self._max_rotor_speed = max_rotor_speed
        self._rotor_speed_half_time = rotor_speed_half_time

        # sanity checks
        assert self.rotational_drag.shape == (3,)
        assert self.translational_drag.shape == (3,)
        assert self.frame_inertia.shape == (3,)
        assert self.gravity.shape == (3,)

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
    def max_rotor_speed(self):
        return self._max_rotor_speed

    @property
    def rotor_speed_half_time(self):
        return self._rotor_speed_half_time

    @property
    def as_tuple(self):
        return (self.thrust_factor, self.drag_factor, self.mass, self.rotational_drag,
                self.translational_drag, self.arm_length, self.rotor_inertia, self.frame_inertia,
                self.gravity, self.max_rotor_speed, self.rotor_speed_half_time)


dyn_state_np = np.dtype([('position', np.float32, 3),
                         ('attitude', np.float32, 3),
                         ('velocity', np.float32, 3),
                         ("rotorspeeds", np.float32, 4),
                         ("desired_speeds", np.float32, 4),
                         ("angular_velocity", np.float32, 3)])


class DynamicsState(object):
    def __init__(self):
        self._position = np.zeros(3)
        self._attitude = Euler(0.0, 0.0, 0.0)
        self._velocity = np.zeros(3)
        self._rotorspeeds = np.zeros(4)
        self._desired_rotor_speeds = np.zeros(4)
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

    @property
    def desired_rotor_speeds(self):
        return self._desired_rotor_speeds

    @desired_rotor_speeds.setter
    def desired_rotor_speeds(self, value):
        self._desired_rotor_speeds[:] = value

    @property
    def angular_velocity(self):
        return self._angular_velocity

    @property
    def net_rotor_speed(self):
        return self._rotorspeeds[0] - self._rotorspeeds[1] + self._rotorspeeds[2] - self._rotorspeeds[3]

    @property
    def as_np(self):
        return np.array((self._position, self._attitude._euler, self._velocity, self._rotorspeeds,
                         self._desired_rotor_speeds, self._angular_velocity), dtype=dyn_state_np)
