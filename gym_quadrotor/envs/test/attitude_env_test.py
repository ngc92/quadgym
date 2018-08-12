import pytest
from gym_quadrotor.dynamics import DynamicsState, Euler
from gym_quadrotor.envs.attitude_env import *


def test_reward_calculation():
    env = CopterStabilizeAttitudeEnv()

    state = DynamicsState()
    state._attitude = Euler(1.0, 0.0, 0.0)
    assert env._calculate_reward(state) == -1

    # check wraparound of yaw coordinate
    state._attitude = Euler(0.0, 0.0, 2*np.pi)
    assert env._calculate_reward(state) == 0
    assert env._correct_counter == 1

    env._error_target = 0.51
    state._attitude = Euler(0.5, 0.0, 0.0)
    assert env._calculate_reward(state) == pytest.approx(-0.25)
    assert env._correct_counter == 2

    env._error_target = 0.51
    state._attitude = Euler(0.6, 0.0, 0.0)
    assert env._calculate_reward(state) == pytest.approx(-0.6*0.6)
    assert env._correct_counter == 0
