from main import main_experiments, main_experiments_parallel, main_experiments_parallel_std
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

    for R in [3, 5, 8]:
        for d in [1, 5, 11, 20, 100, 200]:
            x_star = np.linspace(0, 1, d)  # optimal solution
            x_0 = np.zeros(d)  # initial guess
            for eta in [1e-1,1e-2]:
                main_experiments_parallel(d, n, eta, alpha, x_star, x_0, R, var_epsilon, num_trials)
                main_experiments_parallel_std(d, n, eta, alpha, x_star, x_0, R, var_epsilon, num_trials)

# DONE: Go through Xi's paper; Check how they describe plug-in and batch mean estimator [done]
# DONE: R = 3,5,8 [Done]
# DONE: without replacement experiment [done]
# DONE: Implement Xi's work: warm up/Sensitivity to X_0? [see BM_v2, it seems that the algo is not really sensitive to initialization] [done]
# TODO: How they prove the non-asymptotic polyak averaging case
# TODO: Theory of martingales 5.5.11 (don't understand now.. Too many chapters to read )
# TODO: Shao Jun's work
# TODO: Proof of non-averaging case
# TODO: without replacement proof [counter example]
# TODO: More experiments
# TODO: Extended (real) experiments