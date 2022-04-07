# import packages
import numpy as np
from scipy.stats import norm

# F for linear regression
#   F(x) = \mathbb{E} [1/2 (a^T x - b)^2]
#        = 1/2 (x-x_star)@cov_a@(x-x_star) + var_epsilon
def F_LR(x, cov_a, x_star, var_epsilon):
    return .5 * (x-x_star) @ cov_a @ (x-x_star) + var_epsilon

# SGD original loop (a_n, b_n) iid sample from normal
# rng: random generator
# x_prev: initial guess
# n: number of iterations
def run_SGD_LR_O(seed, x_star, x_prev, n):
    rng = np.random.default_rng(seed)
    d = len(x_prev)
    x_history = []
    a_n_history = rng.normal(0, 1, (n,d))
    epsilon_n_history = rng.normal(0,var_epsilon,n)
    b_n_history = []
    print(f'Seed: [{seed}/100] ...')
    for iter_num in range(n):
        # sample data
        a_n = a_n_history[iter_num,:]
        epsilon_n = epsilon_n_history[iter_num]
        b_n = a_n @ x_star + epsilon_n
        # update learning rate
        eta_n = eta * (1 + iter_num) ** (-alpha)
        # update rule
        x_n = x_prev - eta_n * (a_n @ x_prev - b_n) * a_n
        x_prev = x_n
        # recording
        x_history.append(x_n)
        b_n_history.append(b_n)
        # output every 1000 iter
        # if iter_num % int(n/10) == int(n/10-1):
        #     print(f'Seed: \t [{seed}/100]\t Iter \t[{iter_num + 1}/{n}]\t\t finished')
    x_out = np.mean(x_history, axis=0)
    return x_out, a_n_history, b_n_history

# SGD bootstrap loop
# compute bootstrap confidence interval
def bootstrap_CI(x_0, n, R, a_n_history, b_n_history):
    bootstrap_output_history = []
    rng_b = np.random.default_rng(1)  # random generator for bootstrap experiment
    bootstrap_samples_all = rng_b.integers(0, n, (R,n))  # bootstrap_samples[i] is the index of data for i-th iteration
    for r in range(1, R + 1):
        if r%5==0:
            print(f'---> bootstrap [{r}/{R}] Done')
        # which is selected uniformly from given data
        # SGD on bootstrap samples
        x_prev = x_0
        x_history = []
        bootstrap_samples = bootstrap_samples_all[r-1,:]
        for iter_num in range(n):
            # sample bootstrap data
            a_n = a_n_history[bootstrap_samples[iter_num]]
            b_n = b_n_history[bootstrap_samples[iter_num]]
            # update learning rate
            eta_n = eta * (1 + iter_num) ** (-alpha)
            # update rule
            x_n = x_prev - eta_n * (a_n @ x_prev - b_n) * a_n
            x_prev = x_n
            # recording
            x_history.append(x_n)
            # output every 1000 iter
            # if iter_num % int(n/10) == int(n/10-1):
            #     print(f'R: \t[{r}/{R}]\t Iter \t[{iter_num + 1}/{n}]\t\t finished')
        bootstrap_output_history.append(np.mean(x_history, axis=0))

    # bootstrap true solution
    # x_r is the optimal solution for the bootstrap problem
    # which is obtainable
    # It can be computed by
    # x_r = inv(sum a_i a_i^T) * sum b_i a_i
    A = np.array(a_n_history[:n]).T @ np.array(a_n_history[:n])
    b = np.array(a_n_history[:n]).T @ b_n_history[:n]
    x_r = np.linalg.solve(A, b)

    # Compute Radius of CI
    Z = norm.ppf(0.975)
    d = len(x_r)
    CI_radius = []
    for ii in range(d):
        sigma_hat = np.sqrt(np.sum(np.array(bootstrap_output_history)[:, ii] - x_r[ii]) ** 2 / R)
        radius_d = Z * sigma_hat
        CI_radius.append(radius_d)

    return x_r, CI_radius


if __name__ == '__main__':

    # basic setting
    var_epsilon = 1  # variance for noise in linear regression
    d = 20  # d = 5,20,100,200
    n = int(1e5)  # sample size
    eta = 5e-1
    alpha = 0.501  # step size eta_i = eta * i^{-alpha}
    x_star = np.linspace(0, 1, d)  # optimal solution
    x_0 = np.zeros(d)  # initial guess
    R = 10 # number of bootstrap
    num_trials = 100

    # mean and variance for generating a_i
    # identity covariance matrix case
    #
    # linear regression model:
    # b_i = x_star^\top a_i + \epsilon_i
    mean_a = np.zeros(d)
    cov_a = np.eye(d)
    Asy_cov = np.eye(d)  # asymptotic covariance matrix

    # SGD origial loop
    # set random seed for original samples
    mean_len_history = []
    std_len_history = []
    len_history = []
    cov_history = []
    for seed in range(1,1+num_trials):
        x_out, a_n_history, b_n_history = run_SGD_LR_O(seed, x_star, x_0, n)
        x_r, CI_radius = bootstrap_CI(x_out, n, R, a_n_history, b_n_history)

        mean_Len = np.mean(CI_radius*2)
        std_Len = np.std(CI_radius*2)
        len_history.append(CI_radius*2)
        mean_len_history.append(mean_Len)
        std_len_history.append(std_Len)
        cover = [1 if abs(x_out[ii]-x_r[ii])<= CI_radius[ii] else 0 for ii in range(len(x_out))]
        cov_history.append(cover)

    for seed in range(1,1+num_trials):
        # debug code
        print('*'*20)
        print(f'Len: {mean_len_history[seed-1]} ({std_len_history[seed-1]})')
    print(np.mean(cov_history))
    # import pdb; pdb.set_trace()

    f = open(f'Result_{d}.txt','a')
    f.write('----->\n')
    f.write(f'\t Cov Rate: {np.mean(cov_history)} \t ({np.std(cov_history)}) \tAvg Len: {np.mean(len_history)} \t ({np.std(len_history)}) \n')
    f.write(f'\td: {d} \t n: {n} \t R: {R} \t eta_0: {eta} \t alpha: {alpha} \t # Trials: {num_trials}\n')
    f.close()