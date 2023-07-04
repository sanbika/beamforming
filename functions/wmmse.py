import numpy as np
from channel_init import *
from decimal import Decimal

# v_ik is a MTx1 vector
# u_ik is a MRx1 vector
# H_ik is a MRxMT matrix
# W_ik is a scalar

# def init_channel(Tx, Rx, user_num):
#     h = np.empty((Rx, Tx, user_num), dtype=complex)
#     for user in range(0, user_num):
#         h[:, :, user] = np.random.randn(Rx, Tx) + 1.j * np.random.randn(Rx, Tx)
#     return h

def init_channel(Tx, Rx, user_num):
    # h = init_rician_channel(Tx, user_num)
    h = init_nakagami_channel(Tx, user_num)
    return h

def init_beamforming_vector(Tx, P, user_num):
    # v = np.random.randn(Tx, 1, user_num)
    # norm_v = np.linalg.norm(v)
    # v = v / norm_v * math.sqrt(P)
    v = np.empty((Tx, 1, user_num))
    for user in range(0, user_num):
        vi = np.random.randn(Tx, 1)
        v[:, :, user] = vi / np.sqrt(np.transpose(vi) @ vi)
    return v


def init_weights(user_num):
    weights = np.abs(np.random.randn(user_num))
    return weights


def get_receive_beamformer(h, v, Rx, user_num, noise_var):
    u = np.empty((Rx, 1, user_num), dtype=complex)
    for i in range(0, user_num):
        denominator = noise_var * np.eye(Rx, dtype=complex)
        for j in range(0, user_num):
            denominator += h[:, :, i] @ v[:, :, j] @ np.transpose(v[:, :, j]) @ np.transpose(h[:, :, i]) #Rx*Rx
        u[:, :, i] = np.linalg.inv(denominator) @ h[:, :, i] @ v[:, :, i] #Rx*1
    return u


def get_weights(h, v, u, user_num):
    w = np.empty(user_num)
    for i in range(0, user_num):
        w[i] = np.abs(np.linalg.inv(1 - np.transpose(u[:, :, i]) @ h[:, :, i] @ v[:, :, i]).item().real) #1*1
    return w

def get_mu(h, u, w, user_num, Tx, P):
    temp_1 = 0
    temp_2 = 0
    for i in range(0, user_num):
        temp_1 += np.transpose(np.conjugate(h[:, :, i])) @ u[:, :, i] * w[i] @ np.transpose(np.conjugate(u[:, :, i])) @ h[:, :, i] # Tx*Tx
        temp_2 += np.transpose(np.conjugate(h[:, :, i])) @ u[:, :, i] * w[i] * w[i] @ np.transpose(np.conjugate(u[:, :, i])) @ h[:, :, i] # Tx*Tx
    # np-eig return [value-L, vector-D]
    # matlab-eig return [vector-D, value(diagonal matirx)-L]
    [Lambda, D] = np.linalg.eig(temp_1)
    Lambda = np.diag(Lambda)
    Delta = np.transpose(np.conjugate(D)) @ temp_2 @ D  # Tx*Tx
    mu = [0.01 + j / 1000 for j in range(0, int((100.001 - 0.01) / 0.001), 1)]
    LHS = np.zeros(len(mu))
    for mu_idx in range(0, len(mu)):
        for t in range(0, Tx):
            LHS[mu_idx] += np.real(Delta[t, t] / np.square(Lambda[t, t] + mu[mu_idx]))
    # find good mu so that LHS intercepts with P
    interception = np.real(LHS) - P
    possible_mu = [i for i, v in enumerate(interception) if v >= 0]
    good_mu = -100 # use as default to indicate does not find good mu
    if len(possible_mu) > 0:
        for i in possible_mu:
            good_mu = mu[i]
        mu_diff = LHS[mu.index(good_mu)] - P
        if abs(mu_diff) > Decimal("0.05"):
            good_mu = -100
        else:
            return good_mu, temp_1
    return good_mu, temp_1


def update_beamforming_vector(mu, temp_1, h, u, w, Tx, Rx, user_num):
    v = np.empty((Tx, 1, user_num), dtype=complex)
    denominator = np.linalg.inv(temp_1 + mu * np.eye(Tx))  # Tx*Tx
    for i in range(0, user_num):
        v[:, :, i] = denominator @ np.transpose(np.conjugate(h[:, :, i])) @ u[:, :, i] * w[i] # Tx, 1
    return v


def get_error(w, prev_w, user_num):
    current_sum = 0
    prev_sum = 0
    for i in range(0, user_num):
        current_sum += np.log(w[i])
        prev_sum += np.log(prev_w[i])
    e = np.abs(current_sum - prev_sum)
    return e


def get_sum_rate(h, v, Rx, user_num, noise_var):
    rate = np.empty(user_num)
    for i in range(0, user_num):
        h_h = np.transpose(np.conjugate(h[:, :, i]))  # Tx*Rx
        v_h = np.transpose(np.conjugate(v[:, :, i]))  # 1*Rx
        nominator = h[:, :, i] @ v[:, :, i] @ v_h @ h_h  # Rx*Rx
        denominator = np.eye(Rx, dtype=complex) * noise_var
        for j in range(0, user_num):
            if j != i:
                v_h = np.transpose(np.conjugate(v[:, :, j]))
                denominator += h[:, :, i] @ v[:, :, j] @ v_h @ h_h  # Rx,Rx
        denominator = np.linalg.inv(denominator)
        rate[i] = np.log2(np.real(np.linalg.det(nominator @ denominator + np.eye(Rx, dtype=complex))))
    sum_rate = np.sum(rate)
    return sum_rate, rate


def get_p(v, user_num):
    p = np.empty(user_num)
    for i in range(0, user_num):
        p[i] = np.real(np.transpose(np.conjugate(v[:, :, i])) @ v[:, :, i])
    total_p = np.sum(p)
    return total_p, p

# Tx = 3
# Rx = 1
# P = 10
# user_num = 4
# h = init_channel(Tx, Rx, user_num)
# v = init_beamforming_vector(Tx, P, user_num)
# total_p, p = get_p(v, user_num)
# print(total_p, p)

def wmmse_algorithm(total_num, Tx, Rx, user_num, P, noise_var):
    # current useful data nums
    num = 0

    # construct dataset
    H = np.empty((total_num, Rx, Tx, user_num), dtype=complex)
    W = np.empty((total_num, user_num))
    U = np.empty((total_num, Rx, 1, user_num), dtype=complex)
    MU = np.empty(total_num)
    V = np.empty((total_num, Tx, 1, user_num), dtype=complex)
    SP = np.empty((total_num))
    Pi = np.empty((total_num, user_num))
    SR = np.empty((total_num))
    Ri = np.empty((total_num, user_num))

    # to get the specified quantity of data
    while num < total_num:
        # flag to decide whether the current result is useful
        indicator = True

        # init
        w = init_weights(user_num)
        h = init_channel(Tx, Rx, user_num)
        v = init_beamforming_vector(Tx, P, user_num)

        # stop criterion is error is less than epsilon
        error = 100
        epsilon = 0.001

        # how many iterations has been used in this slice of data, if too many, give up it to save time
        num_iteration = 0

        # when the error does not satisfied and the program does not iterate more than 100 times, and the indicator flag shows continue
        while error > epsilon and num_iteration < 100 and indicator:
            prev_w = w
            u = get_receive_beamformer(h, v, Rx, user_num, noise_var)
            w = get_weights(h, v, u, user_num)
            mu, temp = get_mu(h, u, w, user_num, Tx, P)

            if mu >= 0:
                v = update_beamforming_vector(mu, temp, h, u, w, Tx, Rx, user_num)
                error = get_error(w, prev_w, user_num)
                sr, r = get_sum_rate(h, v, Rx, user_num, noise_var)
                sp, p = get_p(v, user_num)
                num_iteration += 1
                # print('iteration: ', num_iteration, 'wsr: ', sr)
            else:
                indicator = False

        if indicator and num_iteration < 100 and error <= epsilon:
            if num % 10 == 0:
                print(num)
            H[num] = h
            W[num] = w
            U[num] = u
            MU[num] = mu
            V[num] = v
            SP[num] = sp
            Pi[num] = p
            SR[num] = sr
            Ri[num] = r
            num += 1
    return H, W, U, MU, V, SP, Pi, SR, Ri

def wmmse_deepmimo(H, Tx, Rx, user_num, P, noise_var):
    # construct dataset
    total_num = H.shape[0]
    W = np.empty((total_num, user_num))
    U = np.empty((total_num, Rx, 1, user_num), dtype=complex)
    MU = np.empty(total_num)
    V = np.empty((total_num, Tx, 1, user_num), dtype=complex)
    SP = np.empty((total_num))
    Pi = np.empty((total_num, user_num))
    SR = np.empty((total_num))
    Ri = np.empty((total_num, user_num))

    # to get the specified quantity of data
    for i in range(total_num):
        # flag to decide whether the current result is useful
        indicator = True

        # init
        w = init_weights(user_num)
        h = H[i]
        v = init_beamforming_vector(Tx, P, user_num)

        # stop criterion is error is less than epsilon
        error = 100
        epsilon = 0.001

        # how many iterations has been used in this slice of data, if too many, give up it to save time
        num_iteration = 0

        # when the error does not satisfied and the program does not iterate more than 100 times, and the indicator flag shows continue
        while error > epsilon and num_iteration < 100 and indicator:
            prev_w = w
            u = get_receive_beamformer(h, v, Rx, user_num, noise_var)
            w = get_weights(h, v, u, user_num)
            mu, temp = get_mu(h, u, w, user_num, Tx, P)

            if mu >= 0:
                v = update_beamforming_vector(mu, temp, h, u, w, Tx, Rx, user_num)
                error = get_error(w, prev_w, user_num)
                sr, r = get_sum_rate(h, v, Rx, user_num, noise_var)
                sp, p = get_p(v, user_num)
                num_iteration += 1
                # print('iteration: ', num_iteration, 'wsr: ', sr)
            else:
                indicator = False

        if indicator and num_iteration < 100 and error <= epsilon:
            W[i] = w
            U[i] = u
            MU[i] = mu
            V[i] = v
            SP[i] = sp
            Pi[i] = p
            SR[i] = sr
            Ri[i] = r
    return W, U, MU, V, SP, Pi, SR, Ri


# Tx = 3
# Rx = 1
# user_num = 3
# P = 100.0
# noise_var = 1.0
#
# # with open('../datasets/deep_mimo/I1_2p5.npy', 'rb') as f:
# #     H = np.load(f)
#
# with open('../datasets/deep_mimo/I2_28B_1_0.npy', 'rb') as f:
#     H0 = np.load(f)
# with open('../datasets/deep_mimo/I2_28B_1_1.npy', 'rb') as f:
#     H1 = np.load(f)
#
# H = np.concatenate((H0, H1), axis=0)


# W, U, MU, V, SP, Pi, SR, Ri = wmmse_deepmimo(H, Tx, Rx, user_num, P, noise_var)
# print(V)
# print(SP)
# print(SR)

# H, W, U, MU, V, SP, Pi, SR, Ri = wmmse_algorithm(2, Tx, Rx, user_num, P, noise_var)
# print(SR)
# with open('test.npy', 'wb') as f:
#     np.save(f, H)
#     np.save(f, W)
#     np.save(f, U)
#     np.save(f, MU)
#     np.save(f, V)
#     np.save(f, SP)
#     np.save(f, Pi)
#     np.save(f, SR)m
#     np.save(f, Ri)
#
# with open('test.npy', 'rb') as f:
#     H = np.load(f)
#     W = np.load(f)
#     U = np.load(f)
#     MU = np.load(f)
#     V = np.load(f)
#     SP = np.load(f)
#     Pi = np.load(f)
#     SR = np.load(f)
#     Ri = np.load(f)



