import pytest
import mock

from gym_quadrotor.envs.base import *


@pytest.fixture()
def env():
    env = QuadRotorEnvBase()
    env.renderer = mock.Mock()
    env._step_copter = mock.Mock(side_effect=lambda x: (1.0, False, {}))
    env._reset_copter = mock.Mock()
    env._get_state = mock.Mock()
    yield env


def test_clip_attitude():
    state = DynamicsState()
    angvel = np.array([1.0, 0.5, -1.0])
    state._angular_velocity = angvel
    state._attitude = Euler(1.0, 1.0, 1.0)

    def assert_attitude(roll, pitch, yaw):
        assert state.attitude.roll == pytest.approx(roll)
        assert state.attitude.pitch == pytest.approx(pitch)
        assert state.attitude.yaw == pytest.approx(yaw)

    # below threshold: nothing happens
    clip_attitude(state, 1.5)
    assert_attitude(1.0, 1.0, 1.0)
    assert state.angular_velocity == pytest.approx(angvel)

    # yaw exceeds: nothing happens
    state._attitude = Euler(1.0, 1.0, 2.0)

    clip_attitude(state, 1.5)
    assert_attitude(1.0, 1.0, 2.0)
    assert state.angular_velocity == pytest.approx(angvel)

    # roll exceeds
    state._attitude = Euler(2.0, 1.0, 2.0)

    clip_attitude(state, 1.5)
    assert_attitude(1.5, 1.0, 2.0)
    assert state.angular_velocity == pytest.approx(np.zeros(3))

    # pitch exceeds
    state._attitude = Euler(1.0, 2.0, 2.0)

    clip_attitude(state, 1.5)
    assert_attitude(1.0, 1.5, 2.0)
    assert state.angular_velocity == pytest.approx(np.zeros(3))

    # and negative
    state._attitude = Euler(-2.0, -2.0, -2.0)

    clip_attitude(state, 1.5)
    assert_attitude(-1.5, -1.5, -2.0)
    assert state.angular_velocity == pytest.approx(np.zeros(3))


def test_random_angle():
    # check consistency of randomness
    rng = np.random.RandomState(seed=32)
    a1 = random_angle(rng, 90.0)

    rng = np.random.RandomState(seed=32)
    a2 = random_angle(rng, 90.0)

    assert a1.roll == a2.roll
    assert a1.pitch == a2.pitch
    assert a1.yaw == a2.yaw

    rng = mock.Mock()
    a3 = random_angle(rng, 90.0)

    # TODO this assumes the order of the function calls...
    assert rng.uniform.call_args_list[0] == mock.call(low=-np.pi/2, high=np.pi/2)
    assert rng.uniform.call_args_list[1] == mock.call(low=-np.pi/2, high=np.pi/2)
    assert rng.uniform.call_args_list[2] == mock.call(low=-np.pi, high=np.pi)


def test_ensure_fixed_position():
    state = DynamicsState()
    state._position = np.array([1.0, 2.0, 3.0])
    state._velocity = np.array([1.0, 2.0, 3.0])
    ensure_fixed_position(state, 2.0)

    assert state.velocity == pytest.approx(np.zeros(3))
    assert state.position == pytest.approx([0, 0, 2])


def test_base_env_render(env):
    env.render(mode='human', close=False)

    env.renderer.setup.assert_called_once()
    env.renderer.set_center.assert_called_once_with(0.0)
    env.renderer.render.assert_called_once_with('human', False)

    # do not call setup and set_center if closing
    env.render(mode='human', close=True)
    env.renderer.setup.assert_called_once()
    env.renderer.set_center.assert_called_once()

    env.renderer.render.assert_called_with('human', True)


def test_base_env_step(env):
    with pytest.raises(AssertionError):
        env.step([1.0])

    state, reward, done, info = env.step(np.ones(4))
    assert env._state.desired_rotor_speeds == pytest.approx(np.ones(4) * env.setup.max_rotor_speed)

    assert reward == 1.0
    assert done is False
    assert info == {}
    assert state == env._get_state.return_value

    # check action clipping
    env.step(np.ones(4)*3)
    assert env._state.desired_rotor_speeds == pytest.approx(np.ones(4) * env.setup.max_rotor_speed)

    env.step(-np.ones(4) * 3)
    assert env._state.desired_rotor_speeds == pytest.approx(np.zeros(4))


def test_base_env_close(env):
    env.close()
    env.renderer.close.assert_called_once()


def test_base_env_reset(env):
    env.randomize_angle(1)
    new_state = env.reset()

    # check that derived class reset is called
    env._reset_copter.assert_called_once()

    # check that renderer is reset
    env.renderer.set_center.assert_called_once_with(None)

    # check that state has been reset
    assert env.get_copter_state().as_np == DynamicsState().as_np

    assert env._get_state.return_value == new_state
