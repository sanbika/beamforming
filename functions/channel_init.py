import numpy as np
from scipy.stats import nakagami

# Rayleigh fading
def init_rayleigh_channel(Tx, user_num):
    # h = np.random.randn(1, Tx, user_num) + 1j * np.random.randn(1, Tx, user_num)
    x = np.random.normal(loc=0.0, scale=1.0, size=(1, Tx, user_num))
    y = 1j * np.random.normal(loc=0.0, scale=1.0, size=(1, Tx, user_num))
    h = x + 1j * y
    return h

# Rician fading
def init_rician_channel(Tx, user_num):
    x = np.random.normal(loc=1.0, scale=1.0, size=(1, Tx, user_num))
    y = np.random.normal(scale=1.0, size=(1, Tx, user_num))
    h = x + 1j * y
    return h

# Nakagami fading
# nu - shape parameter (4.97 is a random value, this can be adjusted to match the fading conditions of different scenario)
def init_nakagami_channel(Tx, user_num):
    # nu = 4.97
    nu = 10.0
    # nu = 1.0
    # nu = 5.0
    amplitude = nakagami.rvs(nu, size=(1, Tx, user_num))
    x = np.random.normal(scale=1/np.sqrt(2), size=(1, Tx, user_num))
    y = np.random.normal(scale=1/np.sqrt(2), size=(1, Tx, user_num))
    h = amplitude * amplitude * (x + 1j * y)
    return h

# Weibull fading
def init_weibull_channel(Tx, user_num):
    a = 5.
    scale = 2.
    amplitude = np.random.weibull(a, size=(1, Tx, user_num)) * scale
    x = np.random.normal(scale=1/np.sqrt(2), size=(1, Tx, user_num))
    y = np.random.normal(scale=1/np.sqrt(2), size=(1, Tx, user_num))
    h = amplitude * (x + 1j * y)
    return h