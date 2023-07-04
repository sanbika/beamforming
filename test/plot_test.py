import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 20
plt.rcParams["lines.linewidth"] = 3
plt.rcParams["legend.frameon"] = 0
plt.rcParams["axes.labelsize"] = 22

distribution_list = ['nakagami_nu1', 'nakagami_nu5', 'nakagami_nu10', 'rician', 'rayleigh']
data_num_list = ['500', '1k', '2k']
method_list = ['pu', 'pm']
SNR = [10, 20, 30, 40, 50]
data = pd.read_csv("test.csv")
data_num_dict = {
    "500": "1000",
    "1k": "2000",
    "2k": "4000"
}
from functions.colors import *
colors = [purple, red, green, blue]

# same method but different data num
for distribution in distribution_list:
    part_distribution = data.loc[data["test_distribution"] == distribution]
    for method in method_list:
        part_method = part_distribution.loc[part_distribution["method"] == method]
        plt.figure(figsize=(10, 5))
        # y_all = 0
        for idx, data_num in enumerate(data_num_list):
            part_num = part_method.loc[part_method["data_num"] == data_num]
            plt.plot(SNR, part_num["WSR"], label=f"{method}_{data_num_dict[data_num]}", marker='x', color=colors[idx])
            if data_num == '2k':
                plt.plot(SNR, part_num["WMMSE_WSR"], label="wmmse", marker='x', color=colors[-1])
        plt.legend(ncols=2)
        plt.ylabel('WSR')
        plt.xlabel('SNR')
        plt.xticks(SNR)
        plt.tight_layout()
        plt.savefig(f"../figures/{distribution}_{method}_diff_num.png")
        plt.clf()
        plt.close()

# same num but different method
for distribution in distribution_list:
    part_distribution = data.loc[data["test_distribution"] == distribution]
    for data_num in data_num_list:
        part_num = part_distribution.loc[part_distribution["data_num"] == data_num]
        plt.figure(figsize=(10, 5))
        # y_all = 0
        for idx, method in enumerate(method_list):
            part_method = part_num.loc[part_num["method"] == method]
            plt.plot(SNR, part_method["WSR"], label=f"{method}_{data_num_dict[data_num]}", marker='x', color=colors[idx])
            if method == 'pu':
                plt.plot(SNR, part_method["WMMSE_WSR"], label="wmmse", marker='x', color=colors[-1])
        plt.legend(ncol=2)
        plt.ylabel('WSR')
        plt.xlabel('SNR')
        plt.xticks(SNR)
        plt.tight_layout()
        plt.savefig(f"../figures/{distribution}_{data_num}_diff_method.png")
        plt.clf()
        plt.close()

