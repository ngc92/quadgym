import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from collections import deque

from .copter import *


class CopterEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 10
    }

    reward_range = (-1.0, 1.0)

    def __init__(self):
        self.viewer = None
        high = np.array([np.inf]*10)
        
        self.copterparams = CopterParams()
        self.observation_space = spaces.Box(-high, high)
        self.action_space = spaces.Box(-1, 1, (4,))

        self.target         = np.zeros(3)
        self.threshold      =  2 * math.pi / 180
        self.fail_threshold = 15 * math.pi / 180
        self._fail_count    = 0

        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        
        control = np.array(action) * 0.01
        simulate(self.copterstatus, self.copterparams, control, 0.1)

        err = np.max(np.abs(self.copterstatus.attitude - self.target))

        self._steps += 1
        done = bool(self._steps > 100)

        # positive reward for not falling over
        reward = max(0.0, 0.2 * (1 - err / self.fail_threshold))
        if err < self.threshold:
            merr = np.mean(np.abs(self.copterstatus.attitude - self.target)) # this is guaranteed to be smaller than err
            rerr = merr / self.threshold
            reward += 1.1 - rerr

        # reward for keeping velocities low
        velmag = np.mean(np.abs(self.copterstatus.angular_velocity))
        reward += max(0.0, 0.1 - velmag)

        # reward for constant control
        cchange = np.mean(np.abs(control - self._last_control))
        reward += max(0, 0.1 - 2*cchange)

        # normalize reward so that we can get at most 1.0 per step
        reward /= 1.5

        if self.copterstatus.altitude < 0.0 or self.copterstatus.altitude > 10:
            reward = -1
            self._fail_count += 1
            done = True

        # random disturbances
        if self.np_random.rand() < 0.01:
            self.copterstatus.angular_velocity += self.np_random.uniform(low=-10, high=10, size=(3,)) * math.pi / 180

        # change target 
        if self.np_random.rand() < 0.01:
            self.target += self.np_random.uniform(low=-3, high=3, size=(3,)) * math.pi / 180

        self._last_control = control
        return self._get_state(), reward, done, {}

    def _reset(self):
        self.copterstatus = CopterStatus()
        # start in resting position, but with low angular velocity
        self.copterstatus.angular_velocity = self.np_random.uniform(low=-0.1, high=0.1, size=(3,))
        self.copterstatus.velocity         = self.np_random.uniform(low=-0.1, high=0.1, size=(3,))
        self.copterstatus.position         = np.array([0.0, 0, 1])
        self.target = self.np_random.uniform(low=-10, high=10, size=(3,)) * math.pi / 180
        self.copterstatus.attitude = self.target + self.np_random.uniform(low=-5, high=5, size=(3,)) * math.pi / 180
        self._steps = 0
        self._last_control = np.zeros(4)

        return self._get_state()

    def _get_state(self):
        s = self.copterstatus
        # currently, we ignore position and velocity!
        return np.concatenate([s.attitude, s.angular_velocity, self.target, [s.position[2]]])

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-6, 6,-1, 11)
            self.copter_transform = rendering.Transform()
            copter = rendering.make_circle(.1)
            copter.set_color(0,0,0)
            copter.add_attr(self.copter_transform)
            self.viewer.add_geom(copter)

        self.copter_transform.set_translation(self.copterstatus.position[0], self.copterstatus.altitude)
        self.viewer.draw_line((-10, 0.0), (10, 0.0))
        return self.viewer.render(return_rgb_array = mode=='rgb_array')
