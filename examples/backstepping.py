from time import sleep

import gym_quadrotor
from gym_quadrotor.control import calculate_smc
from gym_quadrotor.control.pid import PIDControl
from gym_quadrotor.control.utilities import attitude_to_motor_control
from gym_quadrotor.dynamics import Euler, CopterParams


class BacksteppingController(object):
    def __init__(self):
        p = 15.0
        k = 0.2
        self.P = [p, p, p]
        self.K = [k, k, k]

    def __call__(self, state, target, time):
        state_batch = state[None, :]
        roll_ctrl, pitch_ctrl, yaw_ctrl = calculate_smc(state_batch, 0, CopterParams(), self.K, self.P)

        return attitude_to_motor_control(2.0, roll_ctrl[0], pitch_ctrl[0], yaw_ctrl[0])


import gym
import numpy as np

if __name__ == "__main__":
    env = gym.make("QuadrotorStabilizeAttitude-MotorCommands-v0")
    controller = BacksteppingController()
    target = Euler(0.0, 0.0, 0.0)
    state = env.reset()
    #env.unwrapped.get_copter_state().attitude.yaw = 0.0
    #env.unwrapped.get_copter_state().attitude.roll = 0.0
    #env.unwrapped.get_copter_state().attitude.pitch = 0.0
    #env.unwrapped.get_copter_state()._angular_velocity *= 0
    #env.unwrapped.get_copter_state()._angular_velocity[1] = -0.2

    for i in range(1000):
        action = controller(state, target, i / 50.0)
        state, _, _, _ = env.step(action)
        env.render()
        #sleep(.1)

    env.close()
