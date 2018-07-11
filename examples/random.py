import gym
import numpy as np

import gym_quadrotor


env = gym.make("QuadrotorStabilizeAttitude-v0")
env.reset()

for i in range(1000):
    env.step(2*np.random.rand(4))
    env.render()
