from gym_quadrotor.dynamics import CopterParams


def calculate_smc(state, Onet, params: CopterParams, K, P):
    """
    Calculate the sliding mode control signal.
    :param state: The quadrotor state. An array of shape [BATCH, 6], where the first three entries correspond to the \
    attitude, and the second three to the angular velocity.
    :param ONet: The net speed of the rotors. The sum of all rotor speeds.
    :param params: The parameters of the quadcopter.
    :param K: The `K` parameter of the SMC control law. A vector with three entries.
    :param P: The `P` parameter of the SMC control law. A vector with three entries.
    :return: A tuple containing the roll, pitch, yaw commands, each an array of length BATCH.
    """
    Iroll = params.frame_inertia[0]
    Ipitch = params.frame_inertia[1]
    Iyaw = params.frame_inertia[2]

    Jr = params.rotor_inertia
    L = params.arm_length
    b = params.thrust_factor
    d = params.drag_factor

    Kr = params.rotational_drag
    Kroll = Kr[0]
    Kpitch = Kr[1]
    Kyaw = Kr[2]

    # control parameters
    proll = P[0]
    ppitch = P[1]
    pyaw = P[2]

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

    Rc = roll + (proll + kroll) * droll + proll * kroll * roll
    R = -1.0 / L * (Onet * Jr * dpitch + dpitch * dyaw * (Ipitch - Iyaw) - Kroll * droll + Iroll * Rc)

    Pc = pitch + (ppitch + kpitch) * dpitch + ppitch * kpitch * pitch
    P = -1.0 / L * (-Onet * Jr * droll + droll * dyaw * (Iyaw - Iroll) - Kpitch * dpitch + Ipitch * Pc)

    Yc = yaw + (pyaw + kyaw) * dyaw + pyaw * kyaw * yaw
    Y = -b / d * (droll * dpitch * (Iroll - Ipitch) - Kroll * dyaw + Iyaw * Yc)

    return [R, P, Y]
