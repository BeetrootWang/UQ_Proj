from main import main_experiments, main_experiments_parallel
import numpy as np

if __name__ == '__main__':
    # basic setting
    var_epsilon = 1  # variance for noise in linear regression
    # d = 5  # d = 5,20,100,200
    n = int(1e5)  # sample size
    # eta = 1e-3
    alpha = 0.501  # step size eta_i = eta * i^{-alpha}
    # R = 2  # number of bootstrap
    num_trials = 500

    for R in [2]:
        for d in [11]:
            x_star = np.linspace(0, 1, d)  # optimal solution
            x_0 = np.zeros(d)  # initial guess
            for eta in [5e-1,5e-2,5e-3]:
                main_experiments_parallel(d, n, eta, alpha, x_star, x_0, R, var_epsilon, num_trials)