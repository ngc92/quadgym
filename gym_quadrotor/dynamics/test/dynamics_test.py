import numpy as np
import pytest

from gym_quadrotor.dynamics.coordinates import Euler
from gym_quadrotor.dynamics.dynamics import *


@pytest.fixture()
def params():
    cp = CopterParams()
    cp._mass = 1
    cp._thrustfactor = 1
    return cp


@pytest.fixture()
def state():
    state = DynamicsState()
    state._rotorspeeds = np.zeros(4)
    state._attitude = Euler.from_numpy_array(np.random.rand(3))
    state._position = np.random.rand(3)
    return state


##############################################################################
#                   linear dynamics
##############################################################################
def test_gravity_only(params, state):
    # velocity and rotorspeeds need to be zero.
    state._angular_velocity = np.random.rand(3)
    acceleration = linear_dynamics(params, state)

    assert acceleration == pytest.approx([0.0, 0.0, -9.81])


def test_drag_only(params, state):
    state._attitude = Euler.zero()
    params._gravity = np.zeros(3)
    params._translational_drag = np.array([0.1, 0.2, 0.3])

    state._velocity = np.array([1.0, 1.0, 1.0])
    state._angular_velocity = np.random.rand(3)
    acceleration = linear_dynamics(params, state)

    assert acceleration == pytest.approx([-0.1, -0.2, -0.3])

    # check consistency in rotation
    state._attitude = Euler.from_numpy_array(np.random.rand(3))
    state._velocity = body_to_world(state._attitude, [1.0, 1.0, 1.0])
    acceleration = linear_dynamics(params, state)

    assert acceleration == pytest.approx(body_to_world(state._attitude, [-0.1, -0.2, -0.3]))


def test_rotor_thrust(params, state):
    state._rotorspeeds = [2.0, 2.0, 2.0, 2.0]
    params._gravity = [0.0, 0.0, 0.0]

    acceleration = linear_dynamics(params, state)

    assert acceleration == pytest.approx(body_to_world(state._attitude, [0, 0, 4*2.0**2]))


def test_rotor_speed_for_thrust_consistency(params, state):
    state._attitude = Euler.zero()
    rs = rotor_rotation_for_thrust(2.0 / 4, params)
    state._rotorspeeds = [rs, rs, rs, rs]
    params._gravity = [0.0, 0.0, 0.0]

    acceleration = linear_dynamics(params, state)

    assert acceleration == pytest.approx([0, 0, 2.0 / params.mass])


##############################################################################
#                   angular dynamics
##############################################################################
def test_propellers_equal(params, state):
    """
    In this test all propellers rotate equally fast, which should yield a net torque of zero.
    """
    state._rotorspeeds = [5.0, 5.0, 5.0, 5.0]
    pt = propeller_torques(params, state)

    assert pt == pytest.approx([0, 0, 0])


def test_rotational_drag(params, state):
    """
    Check that rotational drag works.
    """
    params._inertia = np.zeros(3)
    state._angular_velocity = np.array([1.0, 2.0, 0.5])
    params._rotational_drag = np.array([1.0, 0.5, 2.0])

    pt = angular_momentum_body_frame(params, state)

    assert pt == pytest.approx([-1, -1, -1])


def test_rotation_integration(params, state):
    """
    Generate a random point in body frame.
    """
    state._attitude = Euler.zero()
    state._angular_velocity = np.random.rand(3)
    dt = 1e-10

    Pb = np.random.rand(3)
    Pw = body_to_world(state.attitude, Pb)

    # now do a rotation. In world frame
    Pwp = Pw + np.cross(Pw, body_to_world(state._attitude, state._angular_velocity) * dt)

    # rotating the body frame
    er = euler_rate(state)
    a2 = state.attitude.rotated(-er * dt)

    # and in world frame
    Pbp = body_to_world(a2, Pb)
    distance = (Pwp - Pbp) / dt

    assert np.linalg.norm(distance) < 1e-5



"""
def test_conservation_angular_momentum(params, state):
    # this should give us torque-free precession

    state._attitude = Euler(0.0, 0.0, 0.0)
    state._angular_velocity = [0.1, 0.0, 2]#np.random.rand(3)
    params._rotational_drag = np.zeros(3)  # no friction. Important for conservation law
    params._inertia = np.array([1.0, 1.0, 3.0])

    L1 = body_to_world(state.attitude, params.frame_inertia * state.angular_velocity)

    # simulate ten second in high resolution
    for i in range(1000):
        simulate_quadrotor(params, state, 1e-2)
        print(body_to_world(state.attitude, params.frame_inertia * state.angular_velocity))

    L2 = body_to_world(state.attitude, params.frame_inertia * state.angular_velocity)
    assert L1 == pytest.approx(L2)

# do one check that investigates consistency in dL/dt and M
"""

# TODO test for negative rotorspeed prevention code