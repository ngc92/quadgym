import math
import numpy as np

import gym
import numpy
from gym import spaces
from gym.envs.classic_control import rendering
from gym.utils import seeding
from gym_quadrotor.envs.copter import CopterSetup, simulate, CopterStatus


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
        copter.set_color(0, 0, 0)
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

    action_space = spaces.Box(0, 1, (4,))

    def __init__(self, tasks = None):
        self.viewer = None
        self.setup = CopterSetup()
        self._seed()
        self._tasks = [] if tasks is None else tasks

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        self._control = np.array(action)
        simulate(self.copterstatus, self.setup, self._control, 0.02)

        for task in self._tasks:
            task.step()

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
        base_state = [s.attitude, s.angular_velocity, s.velocity, [s.position[2]]]
        tasks_states = [task.get_state().flatten() for task in self._tasks]
        return np.concatenate(base_state + tasks_states)

    def _reset(self):
        self.copterstatus = CopterStatus()
        # start in resting position, but with low angular velocity
        self.copterstatus.angular_velocity = self.np_random.uniform(low=-0.1, high=0.1, size=(3,))
        self.copterstatus.velocity         = self.np_random.uniform(low=-0.1, high=0.1, size=(3,))
        self.copterstatus.position         = np.array([0.0, 0.0, self.np_random.uniform(low=1.0, high=9.0)])
        self.copterstatus.attitude         = self.np_random.uniform(low=-5, high=5, size=(3,)) * math.pi / 180
        self.center                        = self.copterstatus.position[0]

        for task in self._tasks:
            task.reset(self.copterstatus)

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

        return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))