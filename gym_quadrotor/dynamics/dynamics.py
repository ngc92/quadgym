import numpy as np

from gym_quadrotor.dynamics.coordinates import body_z, world_to_body_matrix, body_to_world_matrix, body_to_world, \
    angvel_to_euler
from gym_quadrotor.dynamics.copter import CopterParams, DynamicsState


def linear_dynamics(params, state):
    """
    Calculates the linear acceleration of a quadcopter with parameters
    `params` that is currently in the dynamics state `state`.
    :param CopterParams params: Parameters of the quadrotor.
    :param DynamicsState state: Current dynamics state.
    :return: Linear acceleration in world frame.
    """
    m = params.mass
    b = params.thrust_factor
    Kt = params.translational_drag
    O = state.rotor_speeds
    n = state.attitude
    v = state.velocity

    thrust = b/m * (O[0]**2 + O[1]**2 + O[2]**2 + O[3]**2) * body_z(n)

    Ktw = np.dot(body_to_world_matrix(n), np.dot(np.diag(Kt), world_to_body_matrix(n)))
    drag = np.dot(Ktw, v) / m

    return thrust - drag + params.gravity


def angular_momentum_body_frame(params, state):
    """
    Calculates the angular acceleration of a quadcopter with parameters
    `params` that is currently in the dynamics state `state`.
    :param CopterParams params: Parameters of the quadrotor.
    :param DynamicsState state: Current dynamics state.
    :return: angular acceleration in body frame.
    """
    L = params.arm_length
    b = params.thrust_factor
    d = params.drag_factor
    J = params.rotor_inertia
    O = state.rotor_speeds
    w = state.angular_velocity
    Kr = params.rotational_drag

    Ot = state.net_blade
    gyro = Ot * J * np.array([w[2], -w[1], 0])
    drag = Kr * w
    motor_torque = O[0]**2 + O[1]**2 + O[2]**2 + O[3]**2
    B = np.array([L*b*(O[4]**2 - O[2]**2), L*b*(O[3]**2 - O[1]**2), d*motor_torque]) - drag + gyro
    return B

#
#  TODO where is the w x (Iw) term????
#


def euler_rate(state):
    """
    Calculates the rate of change in euler angle based on the current
    attitude and angular velocity.
    :param DynamicsState state: Current dynamics state.
    :return: d/dt(attitude)
    """
    return angvel_to_euler(state.attitude, state.angular_velocity)


def simulate_quadrotor(params, state, dt):
    """
    Simulate the dynamics of the quadrotor for the timestep given
    in `dt`.
    :param CopterParams params: Parameters of the quadrotor.
    :param DynamicsState state: Current dynamics state.
    """
    acceleration = linear_dynamics(params, state)
    angular_acc  = angular_momentum_body_frame(params, state) / params.frame_inertia

    state._position += 0.5 * dt * dt * acceleration
    state._velocity += dt * acceleration

    state._angular_velocity += dt * angular_acc
    state._attitude.rotate(dt * euler_rate(state))
