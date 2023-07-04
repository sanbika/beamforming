import pandas as pd
from functions.data_processing import *
from functions.beamforming_vector_processing import *
from NN.NNuwmu import *
from time import time


def main(wmmse_data_path, epoches, batch_size, learning_rate, train_ratio, hidden_nodes, Pmax, noise_var,
         user_num, Tx, Rx, split_idx, name_tag):
    # seed = torch.seed()
    # seed 1
    seed = 647936967433600

    torch.manual_seed(seed)

    with open(wmmse_data_path, 'rb') as f:
        H = np.load(f)
        W = np.load(f)
        U = np.load(f)
        MU = np.load(f)
        V = np.load(f)
        SP = np.load(f)
        Pi = np.load(f)
        SR = np.load(f)
        Ri = np.load(f)

    def loss_fn(rate):
        loss = torch.neg(rate)
        return loss

    if torch.cuda.is_available():
        print("Using GPU")
        torch.set_default_tensor_type('torch.cuda.DoubleTensor')

    H = torch.tensor(H)
    H = torch.view_as_real(H)
    SUM_RATE = torch.tensor(SR)

    # Initialize dataset
    train_h, train_r, val_h, val_r = dataset_init(H, SUM_RATE, train_ratio, Tx, Rx, user_num, split_idx)

    # model, optimizer
    model_w = NNw(Tx, Rx, user_num, hidden_nodes)
    model_u = NNu(Tx, Rx, user_num, hidden_nodes)
    model_mu = NNmu(Tx, Rx, user_num, hidden_nodes)

    model_w = model_w.double()
    model_u = model_u.double()
    model_mu = model_mu.double()

    model_w.to(torch.device("cuda"))
    model_u.to(torch.device("cuda"))
    model_mu.to(torch.device("cuda"))

    # learning_rate = 1e-2
    optimizer_w = torch.optim.Adam(model_w.parameters(), lr=learning_rate)
    optimizer_u = torch.optim.Adam(model_u.parameters(), lr=learning_rate)
    optimizer_mu = torch.optim.Adam(model_mu.parameters(), lr=learning_rate)

    # training setting
    train_total_num = train_h.shape[0]
    val_total_num = val_h.shape[0]

    train_iterations = int(train_total_num / batch_size)  # the num of times to update model parameters in one epoch

    # unsupervised training
    start_time = time()
    for epoch in range(1, epoches + 1):
        # shuffle dataset at the start of every epoch
        if epoch != 1:
            train_h, train_r = shuffle_data(train_h, train_r, Tx, Rx, user_num)

        # tensors to store predicted power allocation and rate per user in each epoch
        train_pred_r = torch.empty(train_total_num)
        val_pred_r = torch.empty(val_total_num)

        train_pred_ri = torch.empty((train_total_num, user_num))
        val_pred_ri = torch.empty((val_total_num, user_num))

        train_pred_p = torch.empty((train_total_num, user_num))
        val_pred_p = torch.empty((val_total_num, user_num))

        # train
        for iteration in range(0, train_iterations):
            for i in range(batch_size * iteration, batch_size * (iteration + 1)):
                # get predictions
                pred_w = model_w(train_h[i])
                pred_u = model_u(train_h[i])
                pred_mu = model_mu(train_h[i])

                # processing format of predicted w, u, mu
                pred_w = torch.squeeze(pred_w, 0)
                pred_u = pred_u.reshape(Rx, 1, user_num, 2)
                pred_mu = torch.squeeze(pred_mu, 0)

                # reconstruct beamforming vector
                pred_v = beamforming_uwv(train_h[i], pred_u, pred_w, pred_mu, Tx, Rx, user_num, noise_var)

                # normalize p,v of uwmu
                pred_p = calculate_pi(pred_v, user_num)
                if torch.sum(pred_p) > Pmax:
                    pred_v = normalize_beamforming_vector(pred_v, Pmax)
                    pred_p = calculate_pi(pred_v, user_num)

                # store power allocation
                train_pred_p[i] = pred_p

                # calculate sum rate
                pred_r, ri = calculate_sum_rate(train_h[i], pred_v, Tx, Rx, user_num, noise_var)
                train_pred_r[i] = pred_r
                train_pred_ri[i] = ri

                # get loss and calculate gradients
                loss = loss_fn(pred_r)
                loss.backward()

            # update NN paras
            optimizer_w.step()
            optimizer_u.step()
            optimizer_mu.step()
            optimizer_w.zero_grad()
            optimizer_u.zero_grad()
            optimizer_mu.zero_grad()

        # validation
        with torch.no_grad():
            # get predictions
            for i in range(val_total_num):
                pred_w = model_w(val_h[i])
                pred_u = model_u(val_h[i])
                pred_mu = model_mu(val_h[i])

                # scale p,l and processing u,w,mu formats
                pred_w = torch.squeeze(pred_w, 0)
                pred_u = pred_u.reshape(Rx, 1, user_num, 2)
                pred_mu = torch.squeeze(pred_mu, 0)

                # reconstruct beamforming vector
                pred_v = beamforming_uwv(val_h[i], pred_u, pred_w, pred_mu, Tx, Rx, user_num, noise_var)

                # normalize p,v of uwmu
                pred_p = calculate_pi(pred_v, user_num)
                if torch.sum(pred_p) > Pmax:
                    pred_v = normalize_beamforming_vector(pred_v, Pmax)
                    pred_p = calculate_pi(pred_v, user_num)

                # store power allocation
                val_pred_p[i] = pred_p

                # calculate sum rate
                pred_r, ri = calculate_sum_rate(val_h[i], pred_v, Tx, Rx, user_num, noise_var)
                val_pred_r[i] = pred_r
                val_pred_ri[i] = ri

            print(f"Epoch {epoch}:"
                  f"Rate-Tr {torch.mean(train_pred_r).item():.2f},"
                  f"Rate-Val {torch.mean(val_pred_r).item():.2f}")

            # #check batch efficiency
            # RECORDS["batch"].append(batch_size)
            # RECORDS["timestamp"].append(time() - start_time)
            # RECORDS["rate_val"].append(torch.mean(val_pred_r).item())
            # RECORDS['rate_train'].append(torch.mean(val_pred_r).item())

    # store trained weights of models
    torch.save(model_u.state_dict(), f'../paras/b40e30d{name_tag}_pu_u.pth')
    torch.save(model_w.state_dict(), f'../paras/b40e30d{name_tag}_pu_w.pth')
    torch.save(model_mu.state_dict(), f'../paras/b40e30d{name_tag}_pu_mu.pth')


# RECORDS = {
#     "timestamp": [],
#     "rate_val": [],
#     "rate_train": [],
#     "batch": [],
# }

noise_var = 0.003
epoches = 30
learning_rate = 1e-2
batch = 40
hidden_nodes = 64
Pmax = 3.0
user_num = 3
Tx = 3
Rx = 1

data_path = "../datasets/wmmse_diff_fadings/mixed_rayleigh2k_rician2k_SNR30_power3.npy"
split_idx = 2000

# data_path = "./datasets/wmmse_diff_fadings/mixed_rayleigh1k_rician1k_SNR30_power3.npy"
# split_idx = 1000

# train the model with 1000, 2000, 4000 training data respectively
train_ratio_list = [0.25, 0.5, 1.0]
name_tag_list = ['500', '1k', '2k']

for i in range(0, len(train_ratio_list)):
    main(data_path, epoches, batch, learning_rate, train_ratio_list[i], hidden_nodes, Pmax, noise_var, user_num, Tx, Rx, split_idx, name_tag_list[i])
    # pd.DataFrame(RECORDS).to_csv("records/pure_b40e30d2k.csv", index=False)


