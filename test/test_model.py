from NN.NNuwmu import *
from functions.data_processing import *
from functions.beamforming_vector_processing import *
import pandas as pd

if torch.cuda.is_available():
    print("Using GPU")
    device = torch.device("cuda")
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')

Tx = 3
Rx = 1
user_num = 3
hidden_nodes = 64
train_ratio = 1
P = 3.0

def test(noise_var, data_path, start_idx, end_idx, upth, wpth, mupth):
    # load the data
    with open(data_path, 'rb') as f:
        H = np.load(f)
        W = np.load(f)
        U = np.load(f)
        MU = np.load(f)
        V = np.load(f)
        SP = np.load(f)
        Pi = np.load(f)
        SR = np.load(f)
        Ri = np.load(f)

    # take a subset of the data
    H = H[start_idx:end_idx]
    SR = SR[start_idx:end_idx]
    Pi = Pi[start_idx:end_idx]
    Ri = Ri[start_idx:end_idx]

    # channels (H) & wmmse sum rate (SR)
    H = torch.tensor(H)
    H = torch.view_as_real(H)  # Total_num * Rx * Tx * User * double
    SR = torch.tensor(SR)

    # initialize the test set
    h, sr, _, _ = single_distribution_dataset_init(H, SR, train_ratio, Tx, Rx, user_num)

    # initialize the models
    model_w = NNw(Tx, Rx, user_num, hidden_nodes)
    model_u = NNu(Tx, Rx, user_num, hidden_nodes)
    model_mu = NNmu(Tx, Rx, user_num, hidden_nodes)

    # load the trained model parameters
    model_u.load_state_dict(torch.load(upth))
    model_w.load_state_dict(torch.load(wpth))
    model_mu.load_state_dict(torch.load(mupth))

    # initialize tensors for storing sum rate, rate at each user, and power used by each user
    total_num = h.shape[0]
    pred_sr = torch.empty(total_num)
    pred_ri = torch.empty((total_num, user_num))
    pred_p = torch.empty((total_num, user_num))

    # test for each channel
    for i in range(total_num):
        # get the predicted results
        w = model_w(h[i])
        u = model_u(h[i])
        mu = model_mu(h[i])

        # processing the shape
        w = torch.squeeze(w, 0)
        u = u.reshape(Rx, 1, user_num, 2)
        mu = torch.squeeze(mu, 0)

        # reconstruct the beamforming matrix and normalize it
        v = beamforming_uwv(h[i], u, w, mu, Tx, Rx, user_num, noise_var)
        v = normalize_beamforming_vector(v, P)

        # compute the power used by each user
        pi = calculate_pi(v, user_num)

        # compute the sum rate and rate at each user
        r, ri = calculate_sum_rate(h[i], v, Tx, Rx, user_num, noise_var)

        # store the result of the current channel
        pred_sr[i] = r
        pred_ri[i] = ri
        pred_p[i] = pi

    # compute the mean wsr of our model & wmmse algorithm
    mean_wsr = torch.mean(pred_sr).detach().cpu().numpy().item(0)
    mean_wmmse = torch.mean(sr).detach().cpu().numpy().item(0)
    return mean_wsr, mean_wmmse, pred_ri, pred_p, Pi, Ri


RECORDS = {
    "method": [],
    "data_num": [],
    "test_distribution": [],
    "SNR": [],
    "WSR": [],
    "WMMSE_WSR":[],
    "variance": [],
    "WMMSE_variance": []
}


SNR = [10, 20, 30, 40, 50]
noise_list = [0.3, 0.03, 0.003, 0.0003, 0.00003]
distribution_list = ['nakagami_nu1', 'nakagami_nu5', 'nakagami_nu10', 'rician', 'rayleigh']
data_num_list = ['500', '1k', '2k']
seed = 1

for data_num in data_num_list:
    meta_para_root = f'../paras/seed{seed}/b40e50d{data_num}_pm_'
    us_para_root = f'../paras/seed{seed}/b40e30d{data_num}_pu_'

    for j in distribution_list:
        root_path = f'../datasets/wmmse_diff_fadings/{j}_power3_user3_'
        data_path_list = [root_path+'snr10_1000.npy', root_path+'snr20_1000.npy', root_path+'snr30_1000.npy',
                          root_path+'snr40_1000.npy', root_path+'snr50_1000.npy']

        for i in range(len(noise_list)):
            us_mean_wsr, mean_wmmse, us_ri, us_pi, wmmse_pi, wmmse_ri = test(noise_var=noise_list[i],
                                                                                data_path=data_path_list[i],
                                                                                start_idx=0,
                                                                                end_idx=1000,
                                                                                upth=us_para_root+'u.pth',
                                                                                wpth=us_para_root+'w.pth',
                                                                                mupth=us_para_root+'mu.pth')

            meta_mean_wsr, _, meta_ri, meta_pi, _, _ = test(noise_var=noise_list[i],
                                                                          data_path=data_path_list[i],
                                                                          start_idx=0,
                                                                          end_idx=1000,
                                                                          upth=meta_para_root+'u.pth',
                                                                          wpth=meta_para_root+'w.pth',
                                                                          mupth=meta_para_root+'mu.pth')

            # compute vairances
            us_var = torch.var(torch.mean(us_pi, 0)).detach().cpu().numpy().item(0)
            wmmse_var = np.var(np.mean(wmmse_pi, 0))
            meta_var = torch.var(torch.mean(meta_pi, 0)).detach().cpu().numpy().item(0)

            # record unsupervised test result
            RECORDS["method"].append('pu')
            RECORDS["data_num"].append(data_num)
            RECORDS["test_distribution"].append(j)
            RECORDS["SNR"].append(SNR[i])
            RECORDS["WSR"].append(us_mean_wsr)
            RECORDS["WMMSE_WSR"].append(mean_wmmse)
            RECORDS["variance"].append(us_var)
            RECORDS["WMMSE_variance"].append(wmmse_var)

            # record meta test result
            RECORDS["method"].append('pm')
            RECORDS["data_num"].append(data_num)
            RECORDS["test_distribution"].append(j)
            RECORDS["SNR"].append(SNR[i])
            RECORDS["WSR"].append(meta_mean_wsr)
            RECORDS["WMMSE_WSR"].append(mean_wmmse)
            RECORDS["variance"].append(meta_var)
            RECORDS["WMMSE_variance"].append(wmmse_var)

pd.DataFrame(RECORDS).to_csv("test.csv", index=False)

