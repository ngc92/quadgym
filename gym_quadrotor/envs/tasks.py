import math
import numpy as np
from gym_quadrotor.envs import geo


class CopterTask(object):
    def __init__(self, weight=1.0):
        self.has_failed = False
        self.weight = weight

    def reward(self, *args):
        return self._reward(*args) * self.weight

    def _reward(self, *args):
        raise NotImplementedError()

    def reset(self, status):
        self.has_failed = False
        self._reset(status)

    def _reset(self, status):
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
            reward = -10
            self.has_failed = True
        #elif copterstatus.altitude < 0.2 or copterstatus.altitude > 9.8:
        #    reward = -0.1
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

    def _reset(self, status):
        self._last_control = np.zeros(4)


class HoldAngleTask(CopterTask):
    def __init__(self, threshold, fail_threshold, env, **kwargs):
        super(HoldAngleTask, self).__init__(**kwargs)
        self.threshold = threshold
        self.fail_threshold = fail_threshold
        self.env = env

    def _reward(self, copterstatus, control):
        attitude = copterstatus.attitude
        err = np.mean(np.abs(attitude - self.target))
        # positive reward for not falling over
        reward = max(0.0, 0.2 * (1 - err / self.fail_threshold))
        if err < self.threshold:
            merr = np.mean(np.abs(attitude - self.target))  # this is guaranteed to be smaller than err
            rerr = merr / self.threshold
            reward += 1.1 - rerr

        if err > self.fail_threshold:
            reward = -10
            self.has_failed = True

        return reward

    # TODO how do we pass np_random stuff
    def _reset(self, status):
        self.target = self.env.np_random.uniform(low=-10, high=10, size=(3,)) * math.pi / 180

    def step(self):
        # change target
        if self.env.np_random.rand() < 0.01:
            self.target += self.env.np_random.uniform(low=-3, high=3, size=(3,)) * math.pi / 180

    def draw(self, viewer, copterstatus):
        # draw target orientation
        start = (copterstatus.position[0], copterstatus.altitude)
        rotated = np.dot(geo.make_quaternion(self.target[0], self.target[1], self.target[2]).rotation_matrix,
                         [0, 0, 0.5])
        err = np.max(np.abs(copterstatus.attitude - self.target))
        if err < self.fail_threshold:
            color = (0.0, 0.5, 0.0)
        else:
            color = (1.0, 0.0, 0.0)
        viewer.draw_line(start, (start[0]+rotated[0], start[1]+rotated[2]), color=color)

    def get_state(self):
        return np.array([self.target])


class HoverTask(CopterTask):
    def __init__(self, threshold, fail_threshold, **kwargs):
        super(HoverTask, self).__init__(**kwargs)
        self.threshold = threshold
        self.fail_threshold = fail_threshold
        self.target_altitude = 1.0

    def _reward(self, copterstatus, control):
        attitude = copterstatus.attitude
        err = np.mean(np.abs(attitude))
        # positive reward for not falling over
        reward = max(0.0, 1.0 - (err / self.fail_threshold)**2)
        reward += max(0.0, 1.0 - np.mean(copterstatus.velocity**2)) * 0.25
        reward += max(0.0, 1.0 - (copterstatus.altitude - self.target_altitude)**2) * 0.25

        if err > self.fail_threshold:
            reward = -10
            self.has_failed = True

        return reward

    def draw(self, viewer, copterstatus):
        # draw target orientation
        start = (copterstatus.position[0], copterstatus.altitude)
        rotated = np.dot(geo.make_quaternion(0, 0, 0).rotation_matrix,
                         [0, 0, 0.5])
        err = np.mean(np.abs(copterstatus.attitude))
        if err < self.threshold:
            color = (0.0, 0.5, 0.0)
        else:
            color = (1.0, 0.0, 0.0)
        viewer.draw_line(start, (start[0]+rotated[0], start[1]+rotated[2]), color=color)

    def _reset(self, status):
        self.target_altitude = status.altitude
