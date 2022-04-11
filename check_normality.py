# We plot and check if x_out follows normal distribution
from main import run_SGD_LR_O
import numpy as np

if __name__ == '__main__':
    # basic setting
    var_epsilon = 1  # variance for noise in linear regression
    d = 200  # d = 5,20,100,200
    n = int(1e5)  # sample size
    eta = 1e-2
    alpha = 0.501  # step size eta_i = eta * i^{-alpha}
    x_star = np.linspace(0, 1, d)  # optimal solution
    x_0 = np.zeros(d)  # initial guess
    num_trials = 500

    x_out_history = []
    for seed in range(1,1+num_trials):
        x_out, _, _ = run_SGD_LR_O(seed, x_star, x_0, n, eta)
        x_out_history.append(x_out)

    x_out_history = np.array(x_out_history)
    np.save(f'check_norm_data_d_{d}_n_{n}_eta_{eta}.npy', x_out_history)
