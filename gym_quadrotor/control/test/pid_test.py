from gym_quadrotor.control.pid import *
import pytest


def test_pid_init():
    pid = PIDControl(1.0, 2.0, 3.0)

    assert pid.params == pytest.approx([1.0, 2.0, 3.0])


def test_p_only():
    pid = PIDControl(1.0, 0.0, 0.0)

    assert pid(0.0, 2.0, 0.0) == pytest.approx(2.0)
    assert pid(1.0, 1.0, 1.0) == pytest.approx(0.0)
    assert pid(2.0, 4.0, 2.0) == pytest.approx(2.0)


def test_i_only():
    pid = PIDControl(0.0, 1.0, 0.0)

    assert pid(0.0, 2.0, 0.0) == pytest.approx(0.0)
    assert pid(1.0, 1.0, 1.0) == pytest.approx(0.0)
    assert pid(2.0, 4.0, 1.5) == pytest.approx(1.0)


def test_d_only():
    pid = PIDControl(0.0, 0.0, 1.0)

    assert pid(0.0, 2.0, 0.0) == pytest.approx(0.0)
    assert pid(1.0, 1.0, 1.0) == pytest.approx(-1.0)
    assert pid(2.0, 4.0, 1.5) == pytest.approx(-2.0)


def test_reset():
    pid = PIDControl(1.0, 2.0, 3.0)

    pid(0.0, 2.0, 0.0)
    pid(1.0, 1.0, 1.0)
    pid(2.0, 4.0, 2.0)

    pid.reset()

    # this is the first time-step again, so no i, d contribution
    assert pid(0.0, 2.0, 4.0) == pytest.approx(2.0)