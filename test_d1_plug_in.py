from main import main_experiments, main_experiments_parallel_plug_in
import numpy as np

if __name__ == '__main__':
    # basic setting
    var_epsilon = 1  # variance for noise in linear regression
    d = 11  # d = 5,20,100,200
    n = int(1e4)  # sample size
    eta = 1e-1
    alpha = 0.501  # step size eta_i = eta * i^{-alpha}
    x_star = np.linspace(0, 1, d)  # optimal solution
    x_0 = np.zeros(d)  # initial guess
    num_trials = 500

    main_experiments_parallel_plug_in(d, n, eta, alpha, x_star, x_0, var_epsilon, num_trials)