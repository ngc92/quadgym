from gym import ActionWrapper, ObservationWrapper, Wrapper, spaces
import numpy as np


class IncrementalControlWrapper(Wrapper):
    def __init__(self, env):
        super(IncrementalControlWrapper, self).__init__(env)
        self.action_space = spaces.Box(-np.ones(4), np.ones(4))
        self._last_action = np.zeros(4)

    def _step(self, action):
        action = action + self._last_action

