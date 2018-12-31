from gym_quadrotor.dynamics import CopterParams


def calculate_smc_control(state, Onet, params: CopterParams, K, L):
    """
    Calculate the sliding mode control signal.
    :param state: The quadrotor state. An array of shape [BATCH, 6], where the first three entries correspond to the \
    attitude, and the second three to the angular velocity.
    :param ONet: The net speed of the rotors. The sum of all rotor speeds.
    :param params: The parameters of the quadcopter.
    :param K: The `K` parameter of the SMC control law. A vector with three entries.
    :param L: The lambda parameter of the SMC control law. A vector with three entries.
    :return: A tuple containing the roll, pitch, yaw commands, each an array of length BATCH.
    """
    Iroll = params.frame_inertia[0]
    Ipitch = params.frame_inertia[1]
    Iyaw = params.frame_inertia[2]

    Jr = params.rotor_inertia
    Ll = params.arm_length
    b = params.thrust_factor
    d = params.drag_factor

    Kr = params.rotational_drag
    Kroll = Kr[0]
    Kpitch = Kr[1]
    Kyaw = Kr[2]

    # control parameters
    lroll = L[0]
    lpitch = L[1]
    lyaw = L[2]

    kroll = K[0]
    kpitch = K[1]
    kyaw = K[2]

    # split state
    roll = state[:, 0]
    pitch = state[:, 1]
    yaw = state[:, 2]
    droll = state[:, 3]
    dpitch = state[:, 4]
    dyaw = state[:, 5]

    def sign(x):
        import tensorflow as tf
        import numpy as np
        if isinstance(x, tf.Tensor):
            return tf.sign(x)
        return np.sign(x)

    def smc(lmbda, e, k, edot):
        return lmbda * e + k * sign(edot + lmbda * e)

    Rcomp = -Onet * Jr * dpitch - dpitch * dyaw * (Ipitch - Iyaw) + Kroll * droll
    Pcomp = Onet * Jr * droll - droll * dyaw * (Iyaw - Iroll) + Kpitch * dpitch
    Ycomp = -droll * dpitch * (Iroll - Ipitch) + Kyaw * dyaw

    R = Iroll / Ll * smc(lroll, -roll, kroll, -droll) + Rcomp / Ll
    P = Ipitch / Ll * smc(lpitch, -pitch, kpitch, -dpitch) + Pcomp / Ll
    Y = Iyaw * b / d * smc(lyaw, -yaw, kyaw, -dyaw) + Ycomp * b / d

    return [R, P, Y]
