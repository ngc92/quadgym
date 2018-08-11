import math
import numpy as np
from typing import Optional

import gym
from gym import spaces
from gym.utils import seeding
from gym_quadrotor.dynamics import DynamicsState, CopterParams, simulate_quadrotor


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
        action = np.clip(action, 0.0, 1.0)
        assert action.shape == (4,)

        # set the blade speeds
        self._state.desired_rotor_speeds = action * self.setup.max_rotor_speed
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
    def _step_copter(self, action: np.ndarray):
        raise NotImplementedError()

    def _get_state(self):
        raise NotImplementedError()

    def _reset_copter(self):
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

    def limit_attitude(self, max_angle):
        attitude = self._state.attitude
        if attitude.roll > max_angle:
            attitude.roll = max_angle
            self._state.angular_velocity[:] *= 0
        if attitude.roll < -max_angle:
            attitude.roll = -max_angle
            self._state.angular_velocity[:] *= 0
        if attitude.pitch > max_angle:
            attitude.pitch = max_angle
            self._state.angular_velocity[:] *= 0
        if attitude.pitch < -max_angle:
            attitude.pitch = -max_angle
            self._state.angular_velocity[:] *= 0

    def ensure_fixed_position(self):
        self._state._velocity = np.zeros(3)
        self._state._position = np.array([0.0, 0.0, 1.0])
