import torch
import numpy as np

def get_task_set(support_num, query_num):
    task_path_0 = '../datasets/wmmse_diff_fadings/rayleigh_split2k_SNR30_power3.npy'
    task_path_1 = '../datasets/wmmse_diff_fadings/rician_power3_user3_snr30_2000.npy'

    with open(task_path_0, 'rb') as f:
        H0 = np.load(f)
        W0 = np.load(f)
        U0 = np.load(f)
        MU0 = np.load(f)
        V0 = np.load(f)
        SP0 = np.load(f)
        Pi0 = np.load(f)
        SR0 = np.load(f)
        Ri0 = np.load(f)

    with open(task_path_1, 'rb') as f:
        H1 = np.load(f)
        W1 = np.load(f)
        U1 = np.load(f)
        MU1 = np.load(f)
        V1 = np.load(f)
        SP1 = np.load(f)
        Pi1 = np.load(f)
        SR1 = np.load(f)
        Ri1 = np.load(f)


    # Total_num * Rx * Tx * user * 2
    T0_s = {'H': torch.view_as_real(torch.tensor(H0[0:support_num])), 'noise': 0.003, 'SR': torch.tensor(SR0[0:support_num])}
    T1_s = {'H': torch.view_as_real(torch.tensor(H1[0:support_num])), 'noise': 0.003, 'SR': torch.tensor(SR1[0:support_num])}

    start = support_num
    end = support_num + query_num

    T0_q = {'H': torch.view_as_real(torch.tensor(H0[start:end])), 'noise': 0.003, 'SR': torch.tensor(SR0[start:end])}
    T1_q = {'H': torch.view_as_real(torch.tensor(H1[start:end])), 'noise': 0.003, 'SR': torch.tensor(SR1[start:end])}

    # T0_v = {'H': torch.view_as_real(torch.tensor(H0[end:])), 'noise': 0.003, 'SR': torch.tensor(SR0[end:])}
    # T1_v = {'H': torch.view_as_real(torch.tensor(H1[end:])), 'noise': 0.003, 'SR': torch.tensor(SR1[end:])}

    end = 1000
    val_H = np.concatenate((H0[end:], H1[end:]), axis=0)
    val_SR = np.concatenate((SR0[end:], SR1[end:]), axis=0)
    validation_set = {'H': torch.view_as_real(torch.tensor(val_H)), 'noise': 0.003, 'SR': torch.tensor(val_SR)}

    support_set = [T0_s, T1_s]
    query_set = [T0_q, T1_q]
    # validation_set = [T0_v, T1_v]
    
    return support_set, query_set, validation_set


def sample_task_set(task_set):
    size = len(task_set)
    random_index = torch.randperm(size)
    new_set = []
    for i in range(size):
        new_set.append(task_set[random_index[i]])
    return new_set


def sample_k_data(h, k, Rx, Tx, user_num):
    total_num = h.shape[0]
    random_index = torch.randperm(total_num)
    sample_h = torch.empty((k, Rx, Tx, user_num, 2), dtype=torch.float64)
    for i in range(0, k):
        sample_h[i] = h[random_index[i]]
    return sample_h


def get_beamforming_vector(h, u, w, mu, Tx, Rx, user_num):
    I = torch.eye(Tx)
    V = torch.zeros((Tx, Rx, user_num), dtype=torch.complex128)
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
    return V

def get_sum_rate(h, v, Rx, user_num, noise_var):
    h = torch.view_as_complex(h)  # Rx, Tx, user
    v = torch.view_as_complex(v)  # Tx, 1, user
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

def loss_fn(rate):
    loss = -rate
    return loss

def out_loss(rate):
    loss = -rate
    return loss