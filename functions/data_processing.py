import torch
import numpy as np


def obj2complex(obj):
    complex_array = []
    for o in obj:
        temp_1 = []
        for i in o:
            temp_2 = []
            for j in i:
                temp_2.append(j.astype('complex'))
            temp_1.append(temp_2)
        complex_array.append(temp_1)
    return np.array(complex_array)


def obj2double(obj):
    double_array = []
    for o in obj:
        double_array.append(o.astype('double'))
    return np.array(double_array)

def adap_dataset_init(H, Tx, Rx, user_num, adap_num, val_num):
    total_num = H.shape[0]
    random_index = torch.randperm(total_num)

    adap_h = torch.empty((adap_num, Rx, Tx, user_num, 2), dtype=torch.float64)
    val_h = torch.empty((val_num, Rx, Tx, user_num, 2), dtype=torch.float64)

    for i in range(0, adap_num):
        adap_h[i] = H[random_index[i]]

    for i in range(adap_num, adap_num + val_num):
        val_h[i-adap_num] = H[random_index[i]]

    return adap_h, val_h

# this one is used for processing same distribution set
def single_distribution_dataset_init(H, SUM_RATE, train_ratio, Tx, Rx, user_num):
    total_num = H.shape[0]
    train_num = int(np.floor(train_ratio * total_num))
    random_index = torch.randperm(total_num)

    # train_h = torch.empty((train_num, Tx, Rx, user_num, 2), dtype=torch.float64)
    train_h = torch.empty((train_num, Rx, Tx, user_num, 2), dtype=torch.float64)
    train_r = torch.empty(train_num, dtype=torch.float64)

    # val_h = torch.empty((total_num - train_num, Tx, Rx, user_num, 2), dtype=torch.float64)
    val_h = torch.empty((total_num - train_num, Rx, Tx, user_num, 2), dtype=torch.float64)
    val_r = torch.empty((total_num - train_num), dtype=torch.float64)

    for i in range(0, train_num):
        train_h[i] = H[random_index[i]]
        train_r[i] = SUM_RATE[random_index[i]]

    for i in range(train_num, total_num):
        val_h[i - train_num] = H[random_index[i]]
        val_r[i - train_num] = SUM_RATE[random_index[i]]

    # BEFORE: total num 0, Tx 1, Rx 2, user 3, double 4
    # AFTER: total num 0, double 1, Tx 2, Rx 3, user 4
    # train_h = train_h.permute(0, 4, 1, 2, 3)
    # val_h = val_h.permute(0, 4, 1, 2, 3)

    return train_h, train_r, val_h, val_r


# this one is for processing with two distributions
def dataset_init(H, SUM_RATE, train_ratio, Tx, Rx, user_num, split_idx):
    H_0 = H[:split_idx]
    H_1 = H[split_idx:]
    SR_0 = SUM_RATE[:split_idx]
    SR_1 = SUM_RATE[split_idx:]

    total_num_0 = H_0.shape[0]
    total_num_1 = H_1.shape[0]

    train_num_0 = int(np.floor(train_ratio * total_num_0))
    train_num_1 = int(np.floor(train_ratio * total_num_1))
    train_num = train_num_0 + train_num_1

    val_num_0 = total_num_0 - train_num_0
    val_num_1 = total_num_1 - train_num_1
    val_num = val_num_0 + val_num_1

    random_index_0 = torch.randperm(total_num_0)
    train_random_idx_0 = random_index_0[:train_num_0]
    val_random_idx_0 = random_index_0[train_num_0:]
    random_index_1 = torch.randperm(total_num_1)
    train_random_idx_1 = random_index_1[:train_num_1]
    val_random_idx_1 = random_index_1[train_num_1:]

    train_h = torch.empty((train_num, Rx, Tx, user_num, 2), dtype=torch.float64)
    train_r = torch.empty(train_num, dtype=torch.float64)

    val_h = torch.empty((val_num, Rx, Tx, user_num, 2), dtype=torch.float64)
    val_r = torch.empty(val_num, dtype=torch.float64)

    # [0, train_num_0, total_num_0(split_idx), train_num_1, total_num_1(total_num)]
    for i in range(0, train_num_0):
        train_h[i] = H_0[train_random_idx_0[i]]
        train_r[i] = SR_0[train_random_idx_0[i]]
    for i in range(0, train_num_1):
        train_h[i + train_num_0] = H_1[train_random_idx_1[i]]
        train_r[i + train_num_0] = SR_1[train_random_idx_1[i]]

    for i in range(0, val_num_0):
        val_h[i] = H_0[val_random_idx_0[i]]
        val_r[i] = SR_0[val_random_idx_0[i]]
    for i in range(0, val_num_1):
        val_h[i + val_num_0] = H_1[val_random_idx_1[i]]
        val_r[i + val_num_0] = SR_1[val_random_idx_1[i]]

    return train_h, train_r, val_h, val_r

def shuffle_data(h, r, Tx, Rx, user_num):
    total_num = h.shape[0]
    random_index = torch.randperm(total_num)
    # shuffle_h = torch.empty((total_num, 2, Tx, Rx, user_num), dtype=torch.float64)
    shuffle_h = torch.empty((total_num, Rx, Tx, user_num, 2), dtype=torch.float64)
    shuffle_r = torch.empty(total_num, dtype=torch.float64)

    # # total num0, double1, Tx2, Rx3, user4
    # shuffle_h = shuffle_h.permute(0, 4, 1, 2, 3)

    for i in range(0, total_num):
        shuffle_h[i] = h[random_index[i]]
        shuffle_r[i] = r[random_index[i]]

    return shuffle_h, shuffle_r
