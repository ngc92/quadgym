import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from collections import deque

from .copter import *
from . import geo

from gym.envs.classic_control import rendering

def _draw_ground(viewer, center):
    """ helper function that draws the ground indicator.
        The parameter center indicates where the screen center is supposed to be.
    """
    viewer.draw_line((-10+center, 0.0), (10+center, 0.0))
    for i in range(-8, 10, 2):
        pos = round(center / 2) * 2
        viewer.draw_line((pos+i, 0.0), (pos+i-1, -1.0))

def _draw_copter(viewer, setup, status):
    # transformed main axis
    trafo = status.rotation_matrix
    start = (status.position[0], status.altitude)
    def draw_prop(p):
        rotated = np.dot(trafo, setup.propellers[p].position)
        end     = (start[0]+rotated[0], start[1]+rotated[2])
        viewer.draw_line(start, end)
        copter = rendering.make_circle(.1)
        copter.set_color(0,0,0)
        copter.add_attr(rendering.Transform(translation=end))
        viewer.add_onetime(copter)
   
    # draw current orientation
    rotated = np.dot(trafo, [0, 0, 0.5])
    viewer.draw_line(start, (start[0]+rotated[0], start[1]+rotated[2]))

    for i in range(4): draw_prop(i)

class CopterEnvBase(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self, strict_actions=True, tasks = None):
        self.viewer = None
        self.setup = CopterSetup()
        self._seed()
        self._strict_action_space = strict_actions
        self._tasks = [] if tasks is None else tasks

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        if self._strict_action_space:
            assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        
        self._control = self._control_from_action(action)
        for i in range(2):
            simulate(self.copterstatus, self.setup, self._control, 0.01)

        for task in self._tasks: task.step()

        done = False
        reward = 0.0
        for task in self._tasks:
            reward += task.reward(self.copterstatus, self._control)
            done   |= task.has_failed
        
        self._on_step()

        return self._get_state(), reward, done, {"rotor-speed": self.copterstatus.rotor_speeds}

    def _get_state(self):
        s = self.copterstatus
        # currently, we ignore position and velocity!
        base_state = [s.attitude, s.angular_velocity, [s.position[2]]]
        tasks_states = [task.get_state().flatten() for task in self._tasks]
        return np.concatenate(base_state + tasks_states)

    def _reset(self):
        self.copterstatus = CopterStatus()
        # start in resting position, but with low angular velocity
        self.copterstatus.angular_velocity = self.np_random.uniform(low=-0.1, high=0.1, size=(3,))
        self.copterstatus.velocity         = self.np_random.uniform(low=-0.1, high=0.1, size=(3,))
        self.copterstatus.position         = np.array([0.0, 0.0, 1.0])
        self.copterstatus.attitude         = self.np_random.uniform(low=-5, high=5, size=(3,)) * math.pi / 180
        self.center                        = self.copterstatus.position[0]

        for task in self._tasks: task.reset()

        return self._get_state()

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            self.center = self.copterstatus.position[0]
        
        self.center = 0.9*self.center + 0.1*self.copterstatus.position[0]
        self.viewer.set_bounds(-7 + self.center, 7 + self.center,-1, 13)

        
        # draw ground
        _draw_ground(self.viewer, self.center)
        _draw_copter(self.viewer, self.setup, self.copterstatus)

        # finally draw stuff related to the tasks
        for task in self._tasks: task.draw(self.viewer, self.copterstatus)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def _control_from_action(self, action):
        raise NotImplementedError()


class CopterTask(object):
    def __init__(self, weight=1.0):
        self.has_failed = False
        self.weight = weight

    def reward(self, *args):
        return self._reward(*args) * self.weight

    def _reward(self, *args):
        raise NotImplementedError()

    def reset(self):
        self.has_failed = False
        self._reset()

    def _reset(self):
        pass

    def draw(self, *args):
        pass

    def step(self):
        pass

    def get_state(self):
        return np.array([])

class StayAliveTask(CopterTask):
    def __init__(self, **kwargs):
        super(StayAliveTask, self).__init__(**kwargs)

    def _reward(self, copterstatus, control):
        reward = 0
        if copterstatus.altitude < 0.0 or copterstatus.altitude > 10:
            reward = -1
            self.has_failed = True
        return reward

class FlySmoothlyTask(CopterTask):
    def __init__(self, **kwargs):
        super(FlySmoothlyTask, self).__init__(**kwargs)

    def _reward(self, copterstatus, control):
        # reward for keeping velocities low
        velmag = np.mean(np.abs(copterstatus.angular_velocity))
        reward = max(0.0, 0.1 - velmag) 

        # reward for constant control
        cchange = np.mean(np.abs(control - self._last_control))
        reward += max(0, 0.1 - 10*cchange)

        self._last_control = control

        return reward / 0.2  # normed to 1

    def _reset(self):
        self._last_control = np.zeros(4)

class HoldAngleTask(CopterTask):
    def __init__(self, threshold, fail_threshold, env, **kwargs):
        super(HoldAngleTask, self).__init__(**kwargs)
        self.threshold = threshold
        self.fail_threshold = fail_threshold
        self.env = env

    def _reward(self, copterstatus, control):
        attitude = copterstatus.attitude
        err = np.max(np.abs(attitude - self.target))
        # positive reward for not falling over
        reward = max(0.0, 0.2 * (1 - err / self.fail_threshold))
        if err < self.threshold:
            merr = np.mean(np.abs(attitude - self.target)) # this is guaranteed to be smaller than err
            rerr = merr / self.threshold
            reward += 1.1 - rerr

        return reward

    # TODO how do we pass np_random stuff
    def _reset(self):
        self.target = self.env.np_random.uniform(low=-10, high=10, size=(3,)) * math.pi / 180

    def step(self):
        # change target 
        if self.env.np_random.rand() < 0.01:
            self.target += self.env.np_random.uniform(low=-3, high=3, size=(3,)) * math.pi / 180

    def draw(self, viewer, copterstatus):
        # draw target orientation
        start = (copterstatus.position[0], copterstatus.altitude)
        rotated = np.dot(geo.make_quaternion(self.target[0], self.target[1], self.target[2]).rotation_matrix,
                         [0,0,0.5])
        err = np.max(np.abs(copterstatus.attitude - self.target))
        if err < self.fail_threshold:
            color = (0.0, 0.5, 0.0)
        else:
            color = (1.0, 0.0, 0.0)
        viewer.draw_line(start, (start[0]+rotated[0], start[1]+rotated[2]), color=color)

    def get_state(self):
        return np.array([self.target])


class CopterEnv(CopterEnvBase):
    #reward_range = (-1.0, 1.0)

    def __init__(self):
        # prepare the tasks
        stayalive = StayAliveTask(weight = 1.0)
        smooth    = FlySmoothlyTask(weight = 0.2)
        # TODO for now we pass self along to have consistent random
        holdang   = HoldAngleTask(2 * math.pi / 180, 10 * math.pi / 180, self, weight = 1.0)
        super(CopterEnv, self).__init__(tasks = [stayalive, smooth, holdang])

        high = np.array([np.inf]*10)
        
        self.observation_space = spaces.Box(-high, high)
        self.action_space = spaces.Box(0, 1, (4,))
    
    def _on_step(self):
        # random disturbances
        if self.np_random.rand() < 0.01:
            self.copterstatus.rotor_speeds += self.np_random.uniform(low=-2, high=2, size=(4,))

    def _control_from_action(self, action):
        return np.array(action) + 3.3



class CopterEnvEuler(CopterEnvBase):
    def __init__(self):
        # prepare the tasks
        stayalive = StayAliveTask(weight = 1.0)
        smooth    = FlySmoothlyTask(weight = 0.2)
        # TODO for now we pass self along to have consistent random
        holdang   = HoldAngleTask(2 * math.pi / 180, 10 * math.pi / 180, self, weight = 1.0)
        super(CopterEnvEuler, self).__init__(tasks = [stayalive, smooth, holdang])

        high = np.array([np.inf]*10)
        
        self.observation_space = spaces.Box(-high, high)
        self.action_space = spaces.Box(np.array([0.0, -1.0, -1.0, -1.0]), np.ones(4))
    
    def _on_step(self):
        # random disturbances
        if self.np_random.rand() < 0.01:
            self.copterstatus.rotor_speeds += self.np_random.uniform(low=-2, high=2, size=(4,))

    def _control_from_action(self, action):
        # TODO add tests to show that these arguments are ordered correctly
        total = action[0] * 4
        roll  = action[1] / 2   # rotation about x axis
        pitch = action[2] / 2  # rotation about y axis
        yaw   = action[3] / 2
        return coupled_motor_action(total, roll, pitch, yaw) + 3.3

def coupled_motor_action(total, roll, pitch, yaw):
    a = total / 4 - pitch / 2 + yaw / 4
    b = total / 4 + pitch / 2 + yaw / 4
    c = total / 4 + roll  / 2 - yaw / 4
    d = total / 4 - roll  / 2 - yaw / 4
    return np.array([a, b, c, d])
