import torch
import math


def get_scaling(P, Pmax):
    P_norm = torch.norm(P, p=1)
    P_scaling = Pmax / P_norm * P
    return P_scaling


def calculate_pi(v, user_num):
    v = torch.view_as_complex(v)
    p = torch.empty(user_num, dtype=torch.float64)
    for i in range(0, user_num):
        pi = torch.trace(torch.matmul(torch.transpose(torch.conj(v[:, :, i]), 0, 1), v[:, :, i]))
        p[i] = pi.real
    return p


def normalize_beamforming_vector(pred_v, Pmax):
    # pred_v = torch.view_as_complex(pred_v)
    # v = torch.empty((3, 1, 3), dtype=torch.complex128)
    # total_p=0
    # for i in range(user_num):
    #     vi = pred_v[:,:,i]
    #     total_p += torch.conj(torch.transpose(vi,0,1)) @ vi
    # for i in range(user_num):
    #     vi = pred_v[:, :, i]
    #     v[:,:,i] = vi/total_p * math.sqrt(Pmax)
    # v = torch.view_as_real(v)

    norm_v = torch.norm(pred_v)
    pred_v = pred_v / norm_v * math.sqrt(Pmax)

    return pred_v


# def beamforming_pl(h, P, L, Tx, Rx, user_num, noise_var):
#     I = torch.eye(Tx)
#     V = torch.zeros((Tx, Rx, user_num), dtype=torch.complex128)
#     L = torch.squeeze(L, 0)
#     P = torch.squeeze(P, 0)
#     # h = h.permute(1, 2, 3, 0).contiguous()  # Tx, Rx, user, double
#     h = torch.view_as_complex(h)
#     temp = 0
#     for j in range(0, user_num):
#         h_h = torch.transpose(torch.conj(h[:, :, j]), 0, 1)
#         temp = temp + L[j] / noise_var * torch.matmul(h[:, :, j], h_h)
#     for i in range(0, user_num):
#         pi_sqrt = torch.sqrt(P[i])
#         numerator = torch.matmul(torch.linalg.inv(I + temp), h[:, :, i])
#         denominator = torch.norm(torch.matmul(torch.linalg.inv(I + temp), h[:, :, i]), p=2)
#         vi = pi_sqrt * numerator / denominator
#         V[:, :, i] = vi  # Tx, Rx, user
#     V = torch.view_as_real(V)
#     #     V = V.permute(3, 0, 1, 2)  # double, Tx, Rx, user
#     return V


def beamforming_uwv(h, u, w, mu, Tx, Rx, user_num, noise_var):
    I = torch.eye(Tx)
    V = torch.zeros((Tx, Rx, user_num), dtype=torch.complex128)
    # h = h.permute(2, 1, 3, 0).contiguous()  # Rx, Tx, user, double
    h = torch.view_as_complex(h)  # Rx, Tx, user
    u = torch.view_as_complex(u)  # Rx, 1, user
    temp = 0
    for i in range(0, user_num):
        u_h = torch.transpose(torch.conj(u[:, :, i]), 0, 1)  # 1, Rx
        h_h = torch.transpose(torch.conj(h[:, :, i]), 0, 1)  # Tx, Rx
        step1 = torch.matmul(h_h, u[:, :, i])  # Tx*Rx*Rx*1 = Tx*1
        step2 = step1 * w[i]  # Tx, 1
        step3 = torch.matmul(step2, u_h)  # Tx*1*1*Rx = Tx*Rx
        step4 = torch.matmul(step3, h[:, :, i])  # Tx*Rx*Rx*Tx = Tx*Tx
        temp = temp + step4
    temp_inv = torch.linalg.inv(temp + mu * I)
    for i in range(0, user_num):
        h_h = torch.transpose(torch.conj(h[:, :, i]), 0, 1)  # Tx, Rx
        step1 = torch.matmul(temp_inv, h_h)  # Tx*Tx*Tx*Rx = Tx*Rx
        step2 = torch.matmul(step1, u[:, :, i])  # Tx*Rx*Rx*1 = Tx*1
        vi = step2 * w[i]
        V[:, :, i] = vi  # Tx, 1, user
    V = torch.view_as_real(V)  # Tx, 1, user, double
    #     V = V.permute(3, 0, 1, 2)  # double, Tx, Rx, user
    return V


def calculate_sum_rate(h, v, Tx, Rx, user_num, noise_var):
    # h = h.permute(3, 2, 1, 0).contiguous()  # user, Rx, Tx, double
    # v = v.permute(2, 0, 1, 3).contiguous()  # user, Tx, 1, double
    h = torch.view_as_complex(h)  # user, Rx, Tx
    v = torch.view_as_complex(v)  # user, Tx, 1
    r = torch.empty(user_num)
    sum_rate = 0
    for i in range(0, user_num):
        h_h = torch.transpose(torch.conj(h[:, :, i]), 0, 1)  # Tx, Rx
        v_h = torch.transpose(torch.conj(v[:, :, i]), 0, 1)  # 1, Tx
        step1 = torch.matmul(h[:, :, i], v[:, :, i])  # Rx*Tx*Tx*1 = Rx*1
        step2 = torch.matmul(step1, v_h)  # Rx*1*1*Tx = Rx*Tx
        numerator = torch.matmul(step2, h_h)  # Rx*Tx*Tx*Rx = Rx*Rx
        denominator = noise_var * torch.eye(Rx)
        for j in range(0, user_num):
            if i != j:
                v_h = torch.transpose(torch.conj(v[:, :, j]), 0, 1)  # 1, Tx
                step3 = torch.matmul(h[:, :, i], v[:, :, j])  # Rx*Tx*Tx*1 = Rx*1
                step4 = torch.matmul(step3, v_h)  # Rx*1*1*Tx = Rx*Tx
                step5 = torch.matmul(step4, h_h)  # Rx*Tx*Tx*Rx = Rx*Rx
                denominator = denominator + step5
        step6 = numerator * torch.linalg.inv(denominator)  # Rx*Rx
        step7 = torch.eye(Rx) + step6  # Rx*Rx
        r[i] = torch.log2(torch.real(torch.linalg.det(step7)))
        sum_rate = torch.sum(r)
    return sum_rate, r
