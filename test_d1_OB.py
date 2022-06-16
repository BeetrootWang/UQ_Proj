from main import main_experiments_parallel_OBC, main_experiments_parallel_simple_rerun
import numpy as np
# Experiment for Online Bootstrap
if __name__ == '__main__':
    # basic setting
    var_epsilon = 1  # variance for noise in linear regression
    # d = 11  # d = 5,20,100,200
    n = int(1e4)  # sample size
    alpha = 0.501  # step size eta_i = eta * i^{-alpha}
    num_trials = 500

    for B in [3,5,10]:
        for d in [5,20,200]:
            # for eta in np.linspace(0.02,0.1,8):
            for eta in [0.5,0.1,0.05,0.01]:
                for cov_a_str in ['identity','toeplitz','equi']:
                    x_star = np.linspace(0, 1, d)  # optimal solution
                    x_0 = np.zeros(d)  # initial guess
                    # n=int(1e4/B)
                    main_experiments_parallel_simple_rerun(d, n, eta, alpha, x_star, x_0, B, var_epsilon, cov_a_str, num_trials)