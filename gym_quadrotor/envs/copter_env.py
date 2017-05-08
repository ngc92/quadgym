import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from collections import deque

from .copter import *
from . import geo

from gym.envs.classic_control import rendering

class CopterEnvBase(gym.Env):
    def __init__(self):
        self.viewer = None
        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _draw_ground(self, center):
        """ helper function that draws the ground indicator.
            The parameter center indicates where the screen center is supposed to be.
        """
        self.viewer.draw_line((-10+center, 0.0), (10+center, 0.0))
        for i in range(-8, 10, 2):
            pos = round(center / 2) * 2
            self.viewer.draw_line((pos+i, 0.0), (pos+i-1, -1.0))

    def _draw_copter(self, setup, status):
        # transformed main axis
        trafo = status.rotation_matrix
        start = (status.position[0], status.altitude)
        def draw_prop(p):
            rotated = np.dot(trafo, setup.propellers[p].position)
            end     = (start[0]+rotated[0], start[1]+rotated[2])
            self.viewer.draw_line(start, end)
            copter = rendering.make_circle(.1)
            copter.set_color(0,0,0)
            copter.add_attr(rendering.Transform(translation=end))
            self.viewer.add_onetime(copter)
       
        # draw current orientation
        rotated = np.dot(trafo, [0,0,0.5])
        self.viewer.draw_line(start, (start[0]+rotated[0], start[1]+rotated[2]))

        for i in range(4): draw_prop(i)
        
        


class CopterEnv(CopterEnvBase):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 10
    }

    reward_range = (-1.0, 1.0)

    def __init__(self):
        super(CopterEnv, self).__init__()

        high = np.array([np.inf]*10)
        
        self.setup = CopterSetup()
        self.observation_space = spaces.Box(-high, high)
        self.action_space = spaces.Box(0, 1, (4,))

        self.target         = np.zeros(3)
        self.threshold      =  2 * math.pi / 180
        self.fail_threshold = 10 * math.pi / 180
        self._fail_count    = 0
        

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        
        self._control = np.array(action)
        for i in range(10):
            simulate(self.copterstatus, self.setup, self._control, 0.01)
        self._steps += 1

        reward, done = self._calc_reward()

        # random disturbances
        if self.np_random.rand() < 0.01:
            self.copterstatus.angular_velocity += self.np_random.uniform(low=-10, high=10, size=(3,)) * math.pi / 180

        # change target 
        if self.np_random.rand() < 0.01:
            self.target += self.np_random.uniform(low=-3, high=3, size=(3,)) * math.pi / 180

        self._last_control = self._control
        return self._get_state(), reward, done, {}

    def _calc_reward(self):
        err = np.max(np.abs(self.copterstatus.attitude - self.target))
        done = False

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
        cchange = np.mean(np.abs(self._control - self._last_control))
        reward += max(0, 0.1 - 2*cchange)

        # normalize reward so that we can get at most 1.0 per step
        reward /= 1.5

        if self.copterstatus.altitude < 0.0 or self.copterstatus.altitude > 10:
            reward = -1
            self._fail_count += 1
            done = True

        return reward, done

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
        self.center = self.copterstatus.position[0]

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
            self.viewer = rendering.Viewer(500,500)
            self.center = self.copterstatus.position[0]
        
        self.center = 0.9*self.center + 0.1*self.copterstatus.position[0]
        self.viewer.set_bounds(-7 + self.center, 7 + self.center,-1, 13)

        
        # draw ground
        self._draw_ground(self.center)
        self._draw_copter(self.setup, self.copterstatus)
        
        # draw target orientation
        start = (self.copterstatus.position[0], self.copterstatus.altitude)
        rotated = np.dot(geo.make_quaternion(self.target[0], self.target[1], self.target[2]).rotation_matrix,
                         [0,0,0.5])
        err = np.max(np.abs(self.copterstatus.attitude - self.target))
        if err < self.fail_threshold:
            color = (0.0, 0.5, 0.0)
        else:
            color = (1.0, 0.0, 0.0)
        self.viewer.draw_line(start, (start[0]+rotated[0], start[1]+rotated[2]), color=color)
        
        

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
