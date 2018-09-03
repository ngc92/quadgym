from gym_quadrotor.dynamics import Euler
from gym_quadrotor.wrappers.angular_control import attitude_to_motor_control
from gym_quadrotor.dynamics.copter import DynamicsState, CopterParams
from gym_quadrotor.dynamics.dynamics import simulate_quadrotor
import numpy as np
import pytest


@pytest.mark.parametrize("yaw", np.linspace(0, 2*np.pi))
def test_angular_motor_commands(yaw):
    startatt = [0.0, 0.0, yaw]
    check_couple_direction(0, startatt)
    check_couple_direction(1, startatt)
    check_couple_direction(2, startatt)


def check_couple_direction(index, startat = None):
    setup = CopterParams()
    copterstatus = DynamicsState()
    if startat is not None:
        copterstatus._attitude = Euler.from_numpy_array(startat)
    base = np.zeros(3)
    base[index] = 0.25
    control = attitude_to_motor_control(3.0, *base)
    copterstatus.desired_rotor_speeds = control * setup.max_rotor_speed
    copterstatus._rotorspeeds = copterstatus.desired_rotor_speeds
    start_attitude = np.copy(copterstatus.attitude._euler)
    for i in range(10):
        simulate_quadrotor(setup, copterstatus, 0.001)
    delta = copterstatus.attitude._euler - start_attitude
    #assert(delta[index] > 0.0)
    nd = delta / delta[index]
    ref = np.zeros(3)
    ref[index] = 1
    assert abs(nd[0] - ref[0]) < 1e-2
    assert abs(nd[1] - ref[1]) < 1e-2
    assert abs(nd[2] - ref[2]) < 1e-2
