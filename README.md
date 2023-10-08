# Downlink Beamforming Prediction in MISO System using Meta Learning and Unsupervised Learning

Various machine learning methods have been applied in downlink beamforming to yield performance gain.
However, a prevalent limitation among these approaches is their
reliance on substantial quantities of labeled data with high
training complexity, especially when the number of users and
antenna increases. In this paper, we propose two methodologies
designed to mitigate these limitations. Our methods depart
from conventional practices by training models using unlabeled
data while concurrently curtailing the data volume required
for effective training. Moreover, to reduce training complexity,
we adopt an approach, based on the WMMSE and MLBF
algorithm, that decomposes the beamforming vector prediction
into lower-dimensional components. To enhance the adaptivity,
we incorporate Rayleigh and Rician channels during the training.
And we conducted experiments to train models under different
amounts of data and assess model performance under conditions
where testing distributions align with or diverge from the training
distribution. Both methodologies show better performances than
WMMSE at high SNR under the same distribution. Furthermore, models trained through unsupervised learning showcase
heightened generality when confronted with novel environments
characterized by divergent data distributions. Additionally, our
meta learning approach yields commendable performance even
with limited data availability.


## How to Use

To facilitate transparency and reproducibility, all relevant source
code pertaining to dataset generation, training and testing
procedures, and visualization methods are openly accessible
in this repository. 

### Datasets

We generated datasets in `datasets/wmmse_diff_fadings`, the naming pattern in that folder follow the rules:
```
if start with "mixed":
  mixed_{fading1}{number of channels}_{fading2}{number of channels}_SNR{ratio}_power{transmit power setting}.npy
else:
  {fading}_{power}_{number of users}_{snr}_{number of channels}.npy
```

To load data from xxxx.npy, use the following code: 
```python
    with open(data_path, 'rb') as f:
        H = np.load(f) # channel matrix
        W = np.load(f) # user weights
        U = np.load(f) # receive beamforming matrix
        MU = np.load(f) # a Lagrange multiplier attached to the power constraint when finding the first-order optimality condition of the beamforming matrix in WMMSE
        V = np.load(f) # beamforming matrix
        SP = np.load(f) # total power
        Pi = np.load(f) # power for each user
        SR = np.load(f) # sum rate
        Ri = np.load(f) # rate of each user
```

If you wish to generate datasets yourself, we also provide generation code at 'functions/dataset_generation.py'.

By changing line 40, you can alter the generated datasets in terms of the number of transmit antennas, number of receive antennas, number of users, transmit power, and noise. Line 71 specifies the size of the dataset and line 72 sets the number of CPU cores used for dataset generation.

To modify the fading types, please change the functions invoked by 'init_channel' in 'functions/wmmse.py'. For specific fading settings, you can edit them in 'functions/channel_init.py'.


### Training & Testing
Note: Please change the dataset path, parameter paths and results save paths manually.

To train the model, use the following command. And a clearer view of the training logic is provided by the pseudocodes in the paper.


```bash
# Meta-learning
python3 train/meta.py
# Unsupervised-learning
python3 train/unsupervised.py
```

To evaluate the performance of models, use the following commands:

```bash
python3 test/test_model.py
```

You can also view visulazation result by using the following commands:

```bash
python3 test/plot_test.py
```
The output figures will be saved at `figures`.

### Exsisting Results

The `paras` folder contains some existing model parameters trained on different seeds. Corresponding seed values and test results are in subfolders of `seed`.

## Folder tree of this repository is shown below:
```
NN
└─── NNuwmu.py
datasets
└─── wmmse_diff_fadings
  ├─── mixed_rayleigh1k_rician1k_SNR03_power3.npy 
  ├─── nakagami_nu10_power3_user3_snr10_1000.npy
  ...
functions
  ├─── beamforming_vector_processing.py
  ├─── channel_init.py
  ├─── colors.py
  ├─── data_processing.py
  ├─── dataset_generation.py
  ├─── meta_processing.py
  └─── wmmse.py
paras
  ├─── seed1
    └─── {batch number}{epoch number}{half data number}_{training method name}_{output variable}.pth
  ├─── seed2
    └─── {batch number}{epoch number}{half data number}_{training method name}_{output variable}.pth
  ├─── seed3
    └─── {batch number}{epoch number}{half data number}_{training method name}_{output variable}.pth
  ├─── seed4
    └─── {batch number}{epoch number}{half data number}_{training method name}_{output variable}.pth
  └─── seed5
    └─── {batch number}{epoch number}{half data number}_{training method name}_{output variable}.pth
seeds
  ├─── seed1
    ├─── seed.txt
    └─── test.csv
  ├─── seed2
    ├─── seed.txt
    └─── test.csv
  ├─── seed3
    ├─── seed.txt
    └─── test.csv
  ├─── seed4
    ├─── seed.txt
    └─── test.csv
  └─── seed5
    ├─── seed.txt
    └─── test.csv
test
  ├─── plot_test.py
  └─── test_model.py
train
  ├─── meta.py
  └─── unsupervised.py
```
