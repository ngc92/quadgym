import math
import numpy as np
from typing import Optional
import abc

import gym
from gym import spaces
from gym.utils import seeding
from gym_quadrotor.dynamics import DynamicsState, CopterParams, simulate_quadrotor, Euler


class QuadRotorEnvBase(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    action_space = spaces.Box(0, 1, (4,), dtype=np.float32)

    def __init__(self, params: Optional[CopterParams]=None):
        from .rendering import Renderer, Ground, QuadCopter

        # set up the renderer
        self.renderer = Renderer()
        self.renderer.add_object(Ground())
        self.renderer.add_object(QuadCopter(self))

        # set to supplied copter params, or to default value
        if params is None:
            params = CopterParams()
        self.setup = params

        # just initialize the state to default, the rest will be done by reset
        self._state = DynamicsState()
        self.random_state = None
        self.seed()

    # env functions
    def seed(self, seed=None):
        self.random_state, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        action = np.clip(self._process_action(action), 0.0, 1.0)
        assert action.shape == (4,)

        # set the blade speeds. as F ~ w², and we want F ~ action.
        self._state.desired_rotor_speeds = np.sqrt(action) * self.setup.max_rotor_speed
        simulate_quadrotor(self.setup, self._state, 0.02)

        reward, done, info = self._step_copter(action)
        return self._get_state(), reward, done, info

    def render(self, mode='human', close=False):
        if not close:
            self.renderer.setup()

            # update the renderer's center position
            self.renderer.set_center(self._state.position[0])

        return self.renderer.render(mode, close)

    def close(self):
        self.renderer.close()

    def reset(self):
        self._state = DynamicsState()
        self._reset_copter()
        self.renderer.set_center(None)

        return self._get_state()

    def get_copter_state(self):
        return self._state

    # methods to be implemented in derived classes
    @abc.abstractmethod
    def _step_copter(self, action: np.ndarray):
        raise NotImplementedError()

    @abc.abstractmethod
    def _get_state(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def _reset_copter(self):
        raise NotImplementedError()

    def _process_action(self, action):
        return action

    # utility functions
    def randomize_angle(self, max_pitch_roll: float):
        self._state._attitude = random_angle(self.random_state, max_pitch_roll)

    def randomize_velocity(self, max_speed: float):
        self._state.velocity[:] = self.random_state.uniform(low=-max_speed, high=max_speed, size=(3,))

    def randomize_angular_velocity(self, max_speed: float):
        self._state.angular_velocity[:] = self.random_state.uniform(low=-max_speed, high=max_speed, size=(3,))

    def randomize_altitude(self, min_: float, max_: float):
        self._state.position[2] = self.random_state.uniform(low=min_, high=max_)


def clip_attitude(state: DynamicsState, max_angle: float):
    """
    Limits the roll and pitch angle to the given `max_angle`. If roll or pitch exceed that angle,
    they are clipped and the angular velocity is set to 0.
    :param state: The quadcopter state to be modified.
    :param max_angle: Maximum allowed roll and pitch angle.
    :return: nothing.
    """
    attitude = state.attitude
    angular_velocity = state.angular_velocity
    clipped = False

    if attitude.roll > max_angle:
        attitude.roll = max_angle
        angular_velocity[:] = 0
        clipped = True
    if attitude.roll < -max_angle:
        attitude.roll = -max_angle
        angular_velocity[:] = 0
        clipped = True
    if attitude.pitch > max_angle:
        attitude.pitch = max_angle
        angular_velocity[:] = 0
        clipped = True
    if attitude.pitch < -max_angle:
        attitude.pitch = -max_angle
        angular_velocity[:] = 0
        clipped = True
    return clipped


def random_angle(random_state: np.random.RandomState, max_pitch_roll: float):
    """
    Returns a random Euler angle where roll and pitch are limited to [-max_pitch_roll, max_pitch_roll].
    :param random_state: The random state used to generate the random numbers.
    :param max_pitch_roll: Maximum roll/pitch angle, in degrees.
    :return Euler: A new `Euler` object with randomized angles.
    """
    mpr = max_pitch_roll * math.pi / 180

    # small pitch, roll values, random yaw angle
    roll = random_state.uniform(low=-mpr, high=mpr)
    pitch = random_state.uniform(low=-mpr, high=mpr)
    yaw = random_state.uniform(low=-math.pi, high=math.pi)

    return Euler(roll, pitch, yaw)


def ensure_fixed_position(state: DynamicsState, altitude: float = 1.0):
    """
    Changes the state so that the position part is fixed. This resets the linear velocity
    to zero, moves the x and y coordiantes to zero and the z coordinate to the given altitude.
    :param state: State that is manipulated.
    :param altitude: Altitude at which to fix the position. Set to `None` to leave altitude unchanged.
    :return: nothing.
    """
    if altitude is None:
        altitude = state.position[2]
        state._velocity[0] = 0
        state._velocity[1] = 0
    else:
        state._velocity = np.zeros(3)
    state._position = np.array([0.0, 0.0, altitude])


def project_2d(state: DynamicsState):
    """
    Projects all data in  `state` onto the x-z plane.
    :param state:
    :return:
    """
    state.angular_velocity[0] = 0
    state.angular_velocity[2] = 0
    state.velocity[1] = 0
    state.position[1] = 0
    state.attitude.yaw = 0
    state.attitude.roll = 0
