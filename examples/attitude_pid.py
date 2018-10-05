import gym_quadrotor
from gym_quadrotor.control.pid import PIDControl
from gym_quadrotor.control.utilities import attitude_to_motor_control
from gym_quadrotor.dynamics import Euler


class PIDController(object):
    def __init__(self):
        self._roll = PIDControl(3.0, 0.0, 1.0)
        self._pitch = PIDControl(3.0, .0, 1.0)
        self._yaw = PIDControl(.0, .0, .0)

    def __call__(self, state, target, time):
        roll_ctrl = self._roll(state[0], target.roll, time)
        pitch_ctrl = self._pitch(state[1], target.pitch, time)
        yaw_ctrl = self._yaw(state[2], target.yaw, time)

        return attitude_to_motor_control(3.0, roll_ctrl, pitch_ctrl, yaw_ctrl)


import gym
import numpy as np


if __name__ == "__main__":
    env = gym.make("QuadrotorStabilizeAttitude-v0")
    controller = PIDController()
    target = Euler(0.0, 0.0, 0.0)
    state = env.reset()

    for i in range(100):
        action = controller(state, target, i / 50.0)
        state, _, _, _ = env.step(action)
        env.render()

    env.close()
