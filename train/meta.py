import copy
from functions.meta_processing import *
from NN.NNuwmu import *
from functions.beamforming_vector_processing import calculate_pi, normalize_beamforming_vector
import pandas as pd
from time import time



def main(Tx, Rx, user_num, hidden_nodes, epoches, batch, lr_meta, lr_in, support_num, query_num, name_tag):
    # seed 1
    seed = 647936967433600
    torch.manual_seed(seed)

    # start_time = time()
    if torch.cuda.is_available():
        print("Using GPU")
        torch.set_default_tensor_type('torch.cuda.DoubleTensor')

    # init outer NNu, NNmu, NNw
    meta_u = NNu(Tx, Rx, user_num, hidden_nodes)
    meta_w = NNw(Tx, Rx, user_num, hidden_nodes)
    meta_mu = NNmu(Tx, Rx, user_num, hidden_nodes)

    meta_u = meta_u.double()
    meta_w = meta_w.double()
    meta_mu = meta_mu.double()

    meta_u.to(torch.device("cuda"))
    meta_w.to(torch.device("cuda"))
    meta_mu.to(torch.device("cuda"))

    # meta-train
    # init task set
    Ds, Dq, Dv = get_task_set(support_num, query_num)

    for epoch in range(epoches):
        # Randomly change the order of task sequence in support set
        tasks = sample_task_set(Ds)
        # init inner NN with the same weights and bias of outer
        inner_u = copy.deepcopy(meta_u)
        inner_w = copy.deepcopy(meta_w)
        inner_mu = copy.deepcopy(meta_mu)

        optimizer_inner_u = torch.optim.Adam(inner_u.parameters(), lr=lr_in)
        optimizer_inner_w = torch.optim.Adam(inner_w.parameters(), lr=lr_in)
        optimizer_inner_mu = torch.optim.Adam(inner_mu.parameters(), lr=lr_in)

        optimizer_meta_u = torch.optim.Adam(inner_u.parameters(), lr=lr_meta)
        optimizer_meta_w = torch.optim.Adam(inner_w.parameters(), lr=lr_meta)
        optimizer_meta_mu = torch.optim.Adam(inner_mu.parameters(), lr=lr_meta)

        for i in range(len(tasks)):
            support_h = tasks[i]['H']
            noise_var = tasks[i]['noise']

            Pmax = 3.0

            sample_h = sample_k_data(support_h, batch, Rx, Tx, user_num)
            support_rate = torch.empty(batch)
            query_rate = torch.empty(batch)

            for j in range(batch):
                pred_w = inner_w(sample_h[j])
                pred_u = inner_u(sample_h[j])
                pred_mu = inner_mu(sample_h[j])

                pred_w = torch.squeeze(pred_w, 0)
                pred_u = pred_u.reshape(Rx, 1, user_num, 2)
                pred_mu = torch.squeeze(pred_mu, 0)

                pred_v = get_beamforming_vector(sample_h[j], pred_u, pred_w, pred_mu, Tx, Rx, user_num)
                pred_p = calculate_pi(pred_v, user_num)

                if torch.sum(pred_p) > Pmax:
                    pred_v = normalize_beamforming_vector(pred_v, Pmax)

                pred_r, _ = get_sum_rate(sample_h[j], pred_v, Rx, user_num, noise_var)

                support_rate[j] = pred_r
                support_loss = loss_fn(pred_r)
                support_loss.backward()

            optimizer_inner_u.step()
            optimizer_inner_w.step()
            optimizer_inner_mu.step()
            optimizer_inner_u.zero_grad()
            optimizer_inner_w.zero_grad()
            optimizer_inner_mu.zero_grad()

            # query set processing
            # let Dq=Ds and randomly pick a group of data as query data
            query_tasks = tasks
            query_h = query_tasks[i]['H']
            noise_var = query_tasks[i]['noise']

            sample_h = sample_k_data(query_h, batch, Rx, Tx, user_num)

            for j in range(batch):
                pred_w = inner_w(sample_h[j])
                pred_u = inner_u(sample_h[j])
                pred_mu = inner_mu(sample_h[j])

                pred_w = torch.squeeze(pred_w, 0)
                pred_u = pred_u.reshape(Rx, 1, user_num, 2)
                pred_mu = torch.squeeze(pred_mu, 0)

                pred_v = get_beamforming_vector(sample_h[j], pred_u, pred_w, pred_mu, Tx, Rx, user_num)
                pred_p = calculate_pi(pred_v, user_num)

                if torch.sum(pred_p) > Pmax:
                    pred_v = normalize_beamforming_vector(pred_v, Pmax)

                pred_r, _ = get_sum_rate(sample_h[j], pred_v, Rx, user_num, noise_var)

                query_rate[j] = pred_r
                query_loss = out_loss(pred_r)
                query_loss.backward()

        optimizer_meta_u.step()
        optimizer_meta_w.step()
        optimizer_meta_mu.step()
        optimizer_meta_u.zero_grad()
        optimizer_meta_w.zero_grad()
        optimizer_meta_mu.zero_grad()

        meta_u = copy.deepcopy(inner_u)
        meta_w = copy.deepcopy(inner_w)
        meta_mu = copy.deepcopy(inner_mu)

        # validate the result
        val_h = Dv['H']
        val_data_num = val_h.shape[0]
        val_rate = torch.empty(val_data_num)
        noise_var = Dv['noise']

        for j in range(val_data_num):
            pred_w = meta_w(val_h[j])
            pred_u = meta_u(val_h[j])
            pred_mu = meta_mu(val_h[j])

            pred_w = torch.squeeze(pred_w, 0)
            pred_u = pred_u.reshape(Rx, 1, user_num, 2)
            pred_mu = torch.squeeze(pred_mu, 0)

            pred_v = get_beamforming_vector(val_h[j], pred_u, pred_w, pred_mu, Tx, Rx, user_num)
            pred_p = calculate_pi(pred_v, user_num)

            if torch.sum(pred_p) > Pmax:
                pred_v = normalize_beamforming_vector(pred_v, Pmax)

            pred_r, _ = get_sum_rate(val_h[j], pred_v, Rx, user_num, noise_var)
            val_rate[j] = pred_r

        print(f"Epoch {epoch}:"
            f"Val-Rate {torch.mean(val_rate).item():.2f}")

        # RECORDS["timestamp"].append(time() - start_time)
        # RECORDS["batch"].append(batch)
        # RECORDS["rate_val"].append(torch.mean(val_rate).item())


    torch.save(meta_u.state_dict(), f'../paras/b40e50d{name_tag}_pm_u.pth')
    torch.save(meta_w.state_dict(), f'../paras/b40e50d{name_tag}_pm_w.pth')
    torch.save(meta_mu.state_dict(), f'../paras/b40e50d{name_tag}_pm_mu.pth')


# RECORDS = {
#     "timestamp": [],
#     "batch": [],
#     "rate_val": [],
# }

Tx = 3
Rx = 1
user_num = 3
hidden_nodes = 64

epoches = 50
batch = 40
lr_meta = 1e-3
lr_in = 1e-2

# support_num = 2000
query_num = 0 # do not use separate data for query using now

support_num_list = [500, 1000, 2000]
name_tag_list = ['500', '1k', '2k']

for i in range(0, len(support_num_list)):
    main(Tx, Rx, user_num, hidden_nodes, epoches, batch, lr_meta, lr_in, support_num_list[i], query_num, name_tag_list[i])

# pd.DataFrame(RECORDS).to_csv("d2kb40e60_pm.csv", index=False)