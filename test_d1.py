from main import main_experiments, main_experiments_parallel, main_experiments_parallel_std
import numpy as np

if __name__ == '__main__':
    # basic setting
    var_epsilon = 1  # variance for noise in linear regression
    # n = int(1e5)  # sample size
    alpha = 0.501  # step size eta_i = eta * i^{-alpha}
    num_trials = 500

    for R in [10]:
        for d in [5]:
            x_star = np.linspace(0, 1, d)  # optimal solution
            x_0 = np.zeros(d)  # initial guess
            for eta in np.linspace(0.01,0.1,5):
                for cov_a_str in ['identity']:
                    n=int(1e5)
                    main_experiments_parallel(d, n, eta, alpha, x_star, x_0, R, var_epsilon, cov_a_str, num_trials)

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