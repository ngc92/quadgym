import math
import numpy as np
from typing import Optional

import gym
from gym import spaces
from .rendering import Renderer, Ground, QuadCopter
from gym.utils import seeding
from gym_quadrotor.dynamics import DynamicsState, CopterParams, simulate_quadrotor


class QuadRotorEnvBase(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    action_space = spaces.Box(0, 1, (4,), dtype=np.float32)

    def __init__(self, params: Optional[CopterParams]=None):
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

    # env functions
    def seed(self, seed=None):
        self.random_state, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        action = np.asarray(action)
        assert action.shape == (4,)

        # set the blade speeds
        self._state.rotor_speeds[:] = action[:] * self.setup.motor_factor
        simulate_quadrotor(self.setup, self._state, 0.02)

        reward, done, info = self._step(action)
        return self._get_state(), reward, done, info

    def render(self, mode='human', close=False):
        if not close:
            self.renderer.setup()

        # update the renderer's center position
        self.renderer.set_center(self._state.position[0])

        return self.renderer.render(mode, close)

    def reset(self):
        self._state = DynamicsState()
        self._reset()
        self.renderer.set_center(None)

        return self._get_state()

    # methods to be implemented in derived classes
    def _step(self, action: np.ndarray):
        raise NotImplementedError()

    def _get_state(self):
        raise NotImplementedError()

    def _reset(self):
        raise NotImplementedError()

    # utility functions
    def randomize_angle(self, max_pitch_roll: float):
        mpr = max_pitch_roll * math.pi / 180
        # small pitch, roll values, random yaw angle
        self._state.attitude.roll = self.random_state.uniform(low=-mpr, high=mpr)
        self._state.attitude.pitch = self.random_state.uniform(low=-mpr, high=mpr)
        self._state.attitude.yaw = self.random_state.uniform(low=-math.pi, high=math.pi)

    def randomize_velocity(self, max_speed: float):
        self._state.velocity[:] = self.random_state.uniform(low=-max_speed, high=max_speed, size=(3,))

    def randomize_angular_velocity(self, max_speed: float):
        self._state.angular_velocity[:] = self.random_state.uniform(low=-max_speed, high=max_speed, size=(3,))

    def randomize_altitude(self, min_: float, max_: float):
        self._state.position[2] = self.random_state.uniform(low=min_, high=max_)
