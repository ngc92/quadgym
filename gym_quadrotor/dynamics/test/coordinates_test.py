import numpy as np
import pytest
from gym_quadrotor.dynamics.coordinates import *


def base():
    x = np.array([1, 0, 0])
    y = np.array([0, 1, 0])
    z = np.array([0, 0, 1])
    return x, y, z


##########################################################################################
#           direction transformation
##########################################################################################
def test_identity_trafo():
    coincide = Euler(0, 0, 0)
    vec = np.random.rand(3)

    assert body_to_world(coincide, vec) == pytest.approx(vec)
    assert world_to_body(coincide, vec) == pytest.approx(vec)


def test_inversion():
    coincide = Euler.from_numpy_array(np.random.rand(3))
    vec = np.random.rand(3)

    assert world_to_body(coincide, body_to_world(coincide, vec)) == pytest.approx(vec)


def test_pitch():
    x, y, z = base()
    rot = Euler(0, np.pi / 2, 0)

    # x axis is invariant
    assert body_to_world(rot, y) == pytest.approx(y)
    assert world_to_body(rot, y) == pytest.approx(y)

    assert body_to_world(rot, x) == pytest.approx(-z)
    assert world_to_body(rot, x) == pytest.approx(z)

    assert body_to_world(rot, z) == pytest.approx(x)
    assert world_to_body(rot, z) == pytest.approx(-x)


def test_roll():
    x, y, z = base()
    rot = Euler(np.pi / 2, 0, 0)

    # x axis is invariant
    assert body_to_world(rot, x) == pytest.approx(x)
    assert world_to_body(rot, x) == pytest.approx(x)

    # y goes to -z
    assert body_to_world(rot, y) == pytest.approx(z)
    assert world_to_body(rot, y) == pytest.approx(-z)

    # z
    assert body_to_world(rot, z) == pytest.approx(-y)
    assert world_to_body(rot, z) == pytest.approx(y)


def test_yaw():
    x, y, z = base()
    rot = Euler(0, 0, np.pi / 2)

    # z axis is invariant
    assert body_to_world(rot, z) == pytest.approx(z)
    assert world_to_body(rot, z) == pytest.approx(z)

    assert body_to_world(rot, y) == pytest.approx(-x)
    assert world_to_body(rot, y) == pytest.approx(x)

    assert body_to_world(rot, x) == pytest.approx(y)
    assert world_to_body(rot, x) == pytest.approx(-y)


def test_sequence():
    seq = Euler.from_numpy_array(np.random.rand(3))

    wtb_m = world_to_body_matrix(seq)

    # yaw -> pitch -> roll
    wtb_s = world_to_body_matrix(Euler(seq.roll, 0, 0))
    wtb_s = np.dot(wtb_s, world_to_body_matrix(Euler(0, seq.pitch, 0)))
    wtb_s = np.dot(wtb_s, world_to_body_matrix(Euler(0, 0, seq.yaw)))

    assert wtb_m == pytest.approx(wtb_s)


def test_caching():
    seq = Euler.from_numpy_array(np.random.rand(3))
    wtb_1 = world_to_body_matrix(seq)
    wtb_2 = world_to_body_matrix(seq)

    # check that caching reuses the correct thing
    assert wtb_1 == pytest.approx(wtb_2)

    seq.roll = 1.0
    wtb_3 = world_to_body_matrix(seq)
    # check that changed euler angles trigger recalculation
    assert wtb_3 != pytest.approx(wtb_2)


##########################################################################################
#           angular velocity to euler
##########################################################################################

def test_frames_coincide():
    euler = Euler(0, 0, 0)
    x, y, z = base()

    # rotating around x axis == roll
    assert angvel_to_euler(euler, x) == pytest.approx(x)

    # rotating around y axis == pitch
    assert angvel_to_euler(euler, y) == pytest.approx(y)

    # rotating around z axis == yaw
    assert angvel_to_euler(euler, z) == pytest.approx(z)


def test_frames_90deg_roll():
    euler = Euler(np.pi/2, 0, 0)
    x, y, z = base()

    # rotating around x axis == roll, as `euler` also rotates around x''.
    assert angvel_to_euler(euler, x) == pytest.approx(x)

    # rotating around y axis == pitch, with the pi/2 from rotation around x this becomes z
    assert angvel_to_euler(euler, y) == pytest.approx(z)

    # rotating around z axis == yaw
    assert angvel_to_euler(euler, z) == pytest.approx(-y)


def test_frames_90deg_pitch():
    euler = Euler(0, np.pi/2, 0)
    x, y, z = base()

    # rotating around x axis == roll
    assert angvel_to_euler(euler, x) == pytest.approx(x)

    # rotating around y axis == pitch
    assert angvel_to_euler(euler, y) == pytest.approx(y)

    # at pi/2 pitch yaw is no longer defined uniquely, so weird
    # stuff happens here and we cannot test things.


def test_frames_90deg_yaw():
    euler = Euler(0, 0, np.pi/2)
    x, y, z = base()

    # rotating around x axis == roll
    assert angvel_to_euler(euler, x) == pytest.approx(x)

    # rotating around y axis == pitch
    assert angvel_to_euler(euler, y) == pytest.approx(y)

    # rotating around z axis == yaw
    assert angvel_to_euler(euler, z) == pytest.approx(z)
