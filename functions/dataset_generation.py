from multiprocessing.pool import Pool
from functions.wmmse import *


WMMSE_DATA_PATH = "../datasets/"


def process_and_save(data_num, Tx, Rx, user_num, P, noise_var, part_idx):
    print(f"{part_idx} Starts!")
    try:
        H, W, U, MU, V, SP, Pi, SR, Ri = wmmse_algorithm(data_num, Tx, Rx, user_num, P, noise_var)
        path = f"{WMMSE_DATA_PATH}/power{int(P)}_user{user_num}_{data_num}_{part_idx}.npy"
        with open(path, "wb") as f:
            np.save(f, H)
            np.save(f, W)
            np.save(f, U)
            np.save(f, MU)
            np.save(f, V)
            np.save(f, SP)
            np.save(f, Pi)
            np.save(f, SR)
            np.save(f, Ri)
        print(f"{part_idx} Success!")
    except:
        import traceback
        traceback.print_exc()


if __name__ == "__main__":

    # Tx, Rx, user_num, P, noise_var
    params = [
        # (3, 1, 3, 3.0, 0.3)  # SNR 10
        # (3, 1, 3, 3.0, 0.095)  # SNR 15
        # (3, 1, 3, 3.0, 0.03)  # SNR 20
        # (3, 1, 3, 3.0, 0.0095)  # SNR 25
        # (3, 1, 3, 3.0, 0.003) # SNR 30
        # (3, 1, 3, 3.0, 0.00095) # SNR 35
        # (3, 1, 3, 3.0, 0.0003)  # SNR 40
        (3, 1, 3, 3.0, 0.00003)  # SNR 50
        # (3, 1, 3, 3.0, 3.0)  # SNR 0
        # (3, 1, 3, 3.0, 30.0)  # SNR -10

        # (4, 1, 4, 4.0, 0.4)  # SNR 10
        # (4, 1, 4, 4.0, 0.04)  # SNR 20
        # (4, 1, 4, 4.0, 0.004) # SNR 30
        #
        # (3, 1, 3, 10.0, 1.0)  # SNR 10
        # (3, 1, 3, 10.0, 0.3162)  # SNR 15
        # (3, 1, 3, 10.0, 0.1)  # SNR 20
        # (3, 1, 3, 10.0, 0.0316)  # SNR 25
        # (3, 1, 3, 10.0, 0.01) # SNR 30

        # (4, 1, 4, 10.0, 1.0)  # SNR 10
        # (4, 1, 4, 10.0, 0.3162)  # SNR 15
        # (4, 1, 4, 10.0, 0.1)  # SNR 20
        # (4, 1, 4, 10.0, 0.0316)  # SNR 25
        # (4, 1, 4, 10.0, 0.01) # SNR 30
    ]
    for param in params:

        # SNR = 0
        # if str(param[4]) == '1.0':
        #     SNR = 10
        # elif str(param[4]) == '0.1':
        #     SNR = 20
        # elif str(param[4]) == '0.01':
        #     SNR = 30

        files = []
        total_num = 1000
        core_num = 20
        part_size = total_num // core_num
        total_idx = total_num // part_size
        pool = Pool(processes=core_num)
        for idx in range(total_idx):
            # file_name = f"SNR{SNR}_power{int(param[3])}_user{param[2]}_{part_size}_{idx}.npy"
            file_name = f"power{int(param[3])}_user{param[2]}_{part_size}_{idx}.npy"
            func_param = (part_size,) + param + (idx,)
            pool.apply_async(process_and_save, args=func_param)
            files.append(file_name)
        pool.close()
        pool.join()
        data = {
            "H": None,
            "W": None,
            "U": None,
            "MU": None,
            "V": None,
            "SP": None,
            "Pi": None,
            "SR": None,
            "Ri": None,
        }
        for file_name in files:
            with open(WMMSE_DATA_PATH + "/" + file_name, "rb") as f:
                for key in data:
                    if data[key] is None:
                        data[key] = np.load(f)
                    else:
                        data[key] = np.concatenate([data[key], np.load(f)])
        file_name = f"power{int(param[3])}_user{param[2]}_{total_num}.npy"
        # file_name = f"SNR_{SNR}_power{int(param[3])}_user{param[2]}_{total_num}.npy"
        with open(WMMSE_DATA_PATH + "/" + file_name, "wb") as f:
            for key in data:
                np.save(f, data[key])
        print("Data is stored in " + WMMSE_DATA_PATH + "/" + file_name)



