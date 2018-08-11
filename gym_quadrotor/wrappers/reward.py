import gym


class DiscourageLongEpisodesWrapper(gym.RewardWrapper):
    def __init__(self, env, malus_per_step):
        super().__init__(env)
        self._malus_per_step = malus_per_step

    def reward(self, reward):
        return reward - self._malus_per_step

# TODO it would be nice if we still could get the old error.
