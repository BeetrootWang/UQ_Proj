# import packages
import pdb

import numpy as np
from scipy.stats import t, norm
from joblib import Parallel, delayed

# F for linear regression
#   F(x) = \mathbb{E} [1/2 (a^T x - b)^2]
#        = 1/2 (x-x_star)@cov_a@(x-x_star) + var_epsilon
def F_LR(x, cov_a, x_star, var_epsilon):
    return .5 * (x - x_star) @ cov_a @ (x - x_star) + var_epsilon

# SGD original loop (a_n, b_n) iid sample from normal
# rng: random generator
# x_prev: initial guess
# n: number of iterations
def run_SGD_LR_O(seed, x_star, x_prev, n, eta, var_epsilon, mean_a, cov_a, alpha):
    rng = np.random.default_rng(seed)
    d = len(x_prev)
    x_history = []
    # a_n_history = rng.normal(0, 1, (n, d))
    a_n_history = rng.multivariate_normal(mean=mean_a, cov=cov_a, size=(n))
    epsilon_n_history = rng.normal(0, var_epsilon, n)
    b_n_history = []
    for iter_num in range(n):
        # sample data
        a_n = a_n_history[iter_num, :]
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
    # print(x_out)
    return x_out, a_n_history, b_n_history

# SGD bootstrap loop
# compute bootstrap confidence interval
def bootstrap_CI(x_0, n, R, a_n_history, b_n_history, eta, alpha):
    bootstrap_output_history = []
    rng_b = np.random.default_rng(1)  # random generator for bootstrap experiment
    bootstrap_samples_all = rng_b.integers(0, n, (R, n))  # bootstrap_samples[i] is the index of data for i-th iteration
    for r in range(1, R + 1):
        # which is selected uniformly from given data
        # SGD on bootstrap samples
        x_prev = x_0
        x_history = []
        bootstrap_samples = bootstrap_samples_all[r - 1, :]
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

        if r % 5 == 0:
            print(f'---> bootstrap [{r}/{R}] Done')
            # print(np.mean(x_history, axis=0))

    # # bootstrap true solution
    # # x_r is the optimal solution for the bootstrap problem
    # # which is obtainable
    # # It can be computed by
    # # x_r = inv(sum a_i a_i^T) * sum b_i a_i
    # A = np.array(a_n_history[:n]).T @ np.array(a_n_history[:n])
    # b = np.array(a_n_history[:n]).T @ b_n_history[:n]
    # x_r = np.linalg.solve(A, b)

    # Compute Radius of CI
    t_val = t.ppf(0.975, R-1)
    d = np.shape(a_n_history)[1]
    CI_radius = []
    for ii in range(d):
        bar_X_ii = np.mean(np.array(bootstrap_output_history)[:, ii])
        sigma_hat = np.sqrt(np.sum( (np.array(bootstrap_output_history)[:, ii] - bar_X_ii)**2 / (R - 1) ) )
        radius_d = t_val * sigma_hat
        CI_radius.append(radius_d)
    CI_radius = np.array(CI_radius)
    return CI_radius

def bootstrap_CI_wo(x_0, n, R, a_n_history, b_n_history, eta, alpha):
    bootstrap_output_history = []
    rng_b = np.random.default_rng(1)  # random generator for bootstrap experiment
    for r in range(1, R + 1):
        # which is selected uniformly from given data
        # SGD on bootstrap samples
        x_prev = x_0
        x_history = []
        bootstrap_samples = rng_b.permutation(n)
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

        if r % 5 == 0:
            print(f'---> bootstrap [{r}/{R}] Done')
            # print(np.mean(x_history, axis=0))

    # # bootstrap true solution
    # # x_r is the optimal solution for the bootstrap problem
    # # which is obtainable
    # # It can be computed by
    # # x_r = inv(sum a_i a_i^T) * sum b_i a_i
    # A = np.array(a_n_history[:n]).T @ np.array(a_n_history[:n])
    # b = np.array(a_n_history[:n]).T @ b_n_history[:n]
    # x_r = np.linalg.solve(A, b)

    # Compute Radius of CI
    t_val = t.ppf(0.975, R-1)
    d = np.shape(a_n_history)[1]
    CI_radius = []
    for ii in range(d):
        bar_X_ii = np.mean(np.array(bootstrap_output_history)[:, ii])
        sigma_hat = np.sqrt(np.sum( (np.array(bootstrap_output_history)[:, ii] - bar_X_ii)**2 / (R - 1) ) )
        radius_d = t_val * sigma_hat
        CI_radius.append(radius_d)
    CI_radius = np.array(CI_radius)
    return CI_radius

def run_SGD_LR_std_O(seed, x_star, x_prev, n, eta, var_epsilon, mean_a, cov_a, alpha):
    rng = np.random.default_rng(seed)
    d = len(x_prev)
    x_history = []
    # a_n_history = rng.normal(0, 1, (n, d))
    a_n_history = rng.multivariate_normal(mean=mean_a, cov=cov_a, size=(n))
    epsilon_n_history = rng.normal(0, var_epsilon, n)
    b_n_history = []
    for iter_num in range(n):
        # sample data
        a_n = a_n_history[iter_num, :]
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
    x_out = x_n
    # print(x_out)
    return x_out, a_n_history, b_n_history

def run_SGD_LR_plug_in(seed, x_star, x_prev, n, eta, var_epsilon, mean_a, cov_a, alpha, delta=1e-6):
    rng = np.random.default_rng(seed)
    d = len(x_prev)
    x_history = []
    # a_n_history = rng.normal(0, 1, (n, d))
    a_n_history = rng.multivariate_normal(mean=mean_a, cov=cov_a, size=(n))
    epsilon_n_history = rng.normal(0, var_epsilon, n)
    for iter_num in range(n):
        # sample data
        a_n = a_n_history[iter_num, :]
        epsilon_n = epsilon_n_history[iter_num]
        b_n = a_n @ x_star + epsilon_n
        # update learning rate
        eta_n = eta * (1 + iter_num) ** (-alpha)
        # update rule
        x_n = x_prev - eta_n * (a_n @ x_prev - b_n) * a_n
        x_prev = x_n
        # recording
        x_history.append(x_n)
    x_out = np.mean(x_history, axis=0)

    # Compute \tilde A_n and S_n to get sigma hat
    # Use sigma hat to get CI_radius
    epsilon_n_history = np.reshape(epsilon_n_history, (n,1))
    hat_S = (epsilon_n_history * a_n_history).T @ (epsilon_n_history * a_n_history) / n
    hat_A = a_n_history.T @ a_n_history / n
    # import pdb; pdb.set_trace()
    w,V = np.linalg.eig(hat_A)
    W = np.diag(w * (w>delta) + delta * (w<=delta))
    tilde_A_inv = np.linalg.inv(V @ W @ V.T)
    z = norm.ppf(0.975)
    CI_radius = z * np.sqrt(np.diag(tilde_A_inv @ hat_S @ tilde_A_inv)) / np.sqrt(n)
    return x_out, CI_radius

def run_SGD_LR_BM(seed, x_star, x_prev, M, N, n, eta, var_epsilon, mean_a, cov_a, alpha):
    rng = np.random.default_rng(seed)
    d = len(x_prev)
    x_history = []
    # a_n_history = rng.normal(0, 1, (n, d))
    a_n_history = rng.multivariate_normal(mean=mean_a, cov=cov_a, size=(n))
    epsilon_n_history = rng.normal(0, var_epsilon, n)
    for iter_num in range(n):
        # sample data
        a_n = a_n_history[iter_num, :]
        epsilon_n = epsilon_n_history[iter_num]
        b_n = a_n @ x_star + epsilon_n
        # update learning rate
        eta_n = eta * (1 + iter_num) ** (-alpha)
        # update rule
        x_n = x_prev - eta_n * (a_n @ x_prev - b_n) * a_n
        x_prev = x_n
        # recording
        x_history.append(x_n)
    x_out = np.mean(x_history, axis=0)

    # CI for Batch Mean Estimator
    xk = 0
    # import pdb; pdb.set_trace()
    x_bar_M = np.mean(x_history[int(np.floor(N ** (1/(1-alpha))))+1:n] , axis=0)
    BM_Estimator = np.zeros((d,d))
    for k in range(M+1):
        ek = int(np.floor(((k+1)* N) ** (1/(1-alpha))))
        nk = ek - xk
        x_bar_nk = np.mean(x_history[xk:ek+1] , axis=0)
        BM_Estimator += nk * (x_bar_nk - x_bar_M).reshape([d,1]) @ (x_bar_nk - x_bar_M).reshape([1,d]) /M
        xk = ek+1
    z = norm.ppf(0.975)
    CI_radius = z * np.sqrt(np.diag(BM_Estimator))/np.sqrt(n)
    # import pdb; pdb.set_trace()
    return x_out, CI_radius

# SGD bootstrap loop
# compute bootstrap confidence interval
def bootstrap_CI_std(x_0, n, R, a_n_history, b_n_history, eta, alpha):
    bootstrap_output_history = []
    rng_b = np.random.default_rng(1)  # random generator for bootstrap experiment
    bootstrap_samples_all = rng_b.integers(0, n, (R, n))  # bootstrap_samples[i] is the index of data for i-th iteration
    for r in range(1, R + 1):
        # which is selected uniformly from given data
        # SGD on bootstrap samples
        x_prev = x_0
        x_history = []
        bootstrap_samples = bootstrap_samples_all[r - 1, :]
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
        bootstrap_output_history.append(x_n)

        if r % 5 == 0:
            print(f'---> bootstrap [{r}/{R}] Done')
            # print(np.mean(x_history, axis=0))

    # # bootstrap true solution
    # # x_r is the optimal solution for the bootstrap problem
    # # which is obtainable
    # # It can be computed by
    # # x_r = inv(sum a_i a_i^T) * sum b_i a_i
    # A = np.array(a_n_history[:n]).T @ np.array(a_n_history[:n])
    # b = np.array(a_n_history[:n]).T @ b_n_history[:n]
    # x_r = np.linalg.solve(A, b)

    # Compute Radius of CI
    t_val = t.ppf(0.975, R-1)
    d = np.shape(a_n_history)[1]
    CI_radius = []
    for ii in range(d):
        bar_X_ii = np.mean(np.array(bootstrap_output_history)[:, ii])
        sigma_hat = np.sqrt(np.sum( (np.array(bootstrap_output_history)[:, ii] - bar_X_ii)**2 / (R - 1) ) )
        radius_d = t_val * sigma_hat
        CI_radius.append(radius_d)
    CI_radius = np.array(CI_radius)
    return CI_radius

# no longer useful
def main_experiments(d, n, eta, alpha, x_star, x_0, R, var_epsilon, num_trials):
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
    mean_len_history = np.zeros(num_trials)
    std_len_history = np.zeros(num_trials)
    len_history = np.zeros([num_trials,d])
    cov_history = np.zeros([num_trials,d])
    for seed in range(1, 1 + num_trials):
        print(f'Seed: [{seed}/{num_trials}] ...')
        x_out, a_n_history, b_n_history = run_SGD_LR_O(seed, x_star, x_0, n, eta, var_epsilon, alpha)
        CI_radius = bootstrap_CI(x_out, n, R, a_n_history, b_n_history, eta, alpha)

        mean_Len = np.mean(CI_radius * 2)
        std_Len = np.std(CI_radius * 2)
        # import pdb; pdb.set_trace()
        len_history[seed-1,:] = CI_radius*2
        mean_len_history[seed-1] = mean_Len
        std_len_history[seed-1] = std_Len
        cover = [1 if abs(x_out[ii] - x_star[ii]) <= CI_radius[ii] else 0 for ii in range(len(x_out))]
        cov_history[seed-1,:] = cover

    for seed in range(1, 1 + num_trials):
        # debug code
        print('*' * 20)
        print(f'Len: {mean_len_history[seed - 1]:.6f} ({std_len_history[seed - 1]:.6f})')
    print(np.mean(cov_history))
    # import pdb; pdb.set_trace()

    f = open(f'Result_{d}.txt', 'a')
    f.write('----->\n')
    f.write(
        f'\t Cov Rate: {np.mean(cov_history)} \t ({np.std(cov_history)}) \tAvg Len: {np.mean(len_history)} \t ({np.std(len_history)}) \n')
    f.write(f'\td: {d} \t n: {n} \t R: {R} \t eta_0: {eta} \t alpha: {alpha} \t # Trials: {num_trials}\n')
    f.close()

    return

def main_loop(seed, x_star, x_0, n, R, eta, var_epsilon, mean_a, cov_a, alpha, num_trials):
    print(f'Seed: [{seed}/{num_trials}] ...')
    x_out, a_n_history, b_n_history = run_SGD_LR_O(seed, x_star, x_0, n, eta, var_epsilon, mean_a, cov_a, alpha)
    CI_radius = bootstrap_CI(x_0, n, R, a_n_history, b_n_history, eta, alpha)

    mean_Len = np.mean(CI_radius * 2)
    std_Len = np.std(CI_radius * 2)
    cover = [1 if abs(x_out[ii] - x_star[ii]) <= CI_radius[ii] else 0 for ii in range(len(x_out))]

    return mean_Len, std_Len, cover, CI_radius*2, x_out

def main_loop_std(seed, x_star, x_0, n, R, eta, var_epsilon, mean_a, cov_a, alpha, num_trials):
    print(f'Seed: [{seed}/{num_trials}] ...')
    x_out, a_n_history, b_n_history = run_SGD_LR_std_O(seed, x_star, x_0, n, eta, var_epsilon, mean_a, cov_a, alpha)
    CI_radius = bootstrap_CI_std(x_0, n, R, a_n_history, b_n_history, eta, alpha)

    mean_Len = np.mean(CI_radius * 2)
    std_Len = np.std(CI_radius * 2)
    cover = [1 if abs(x_out[ii] - x_star[ii]) <= CI_radius[ii] else 0 for ii in range(len(x_out))]

    return mean_Len, std_Len, cover, CI_radius*2, x_out

def main_loop_plug_in(seed, x_star, x_0, n, eta, var_epsilon, mean_a, cov_a, alpha, num_trials):
    print(f'Seed: [{seed}/{num_trials}] ...')
    x_out, CI_radius = run_SGD_LR_plug_in(seed, x_star, x_0, n, eta, var_epsilon, mean_a, cov_a, alpha)

    mean_Len = np.mean(CI_radius * 2)
    std_Len = np.std(CI_radius * 2)
    cover = [1 if abs(x_out[ii] - x_star[ii]) <= CI_radius[ii] else 0 for ii in range(len(x_out))]

    return mean_Len, std_Len, cover, CI_radius*2, x_out

def main_loop_BM(seed, x_star, x_0, M, N, n, eta, var_epsilon, mean_a, cov_a, alpha, num_trials):
    print(f'Seed: [{seed}/{num_trials}] ...')
    x_out, CI_radius = run_SGD_LR_BM(seed, x_star, x_0, M, N, n, eta, var_epsilon, mean_a, cov_a, alpha)

    mean_Len = np.mean(CI_radius * 2)
    std_Len = np.std(CI_radius * 2)
    cover = [1 if abs(x_out[ii] - x_star[ii]) <= CI_radius[ii] else 0 for ii in range(len(x_out))]

    return mean_Len, std_Len, cover, CI_radius*2, x_out

def main_loop_wo(seed, x_star, x_0, n, R, eta, var_epsilon, mean_a, cov_a, alpha, num_trials):
    print(f'Seed: [{seed}/{num_trials}] ...')
    x_out, a_n_history, b_n_history = run_SGD_LR_O(seed, x_star, x_0, n, eta, var_epsilon, mean_a, cov_a, alpha)
    CI_radius = bootstrap_CI_wo(x_0, n, R, a_n_history, b_n_history, eta, alpha)

    mean_Len = np.mean(CI_radius * 2)
    std_Len = np.std(CI_radius * 2)
    cover = [1 if abs(x_out[ii] - x_star[ii]) <= CI_radius[ii] else 0 for ii in range(len(x_out))]

    return mean_Len, std_Len, cover, CI_radius*2, x_out

def main_experiments_parallel(d, n, eta, alpha, x_star, x_0, R, var_epsilon, cov_a_str, num_trials):
    # mean and variance for generating a_i
    # identity covariance matrix case
    #
    # linear regression model:
    # b_i = x_star^\top a_i + \epsilon_i
    if cov_a_str == 'identity':
        mean_a = np.zeros(d)
        cov_a = np.eye(d)
    elif cov_a_str == 'toeplitz':
        mean_a = np.zeros(d)
        cov_a = np.eye(d)
        r = 0.5
        for ii in range(d):
            for jj in range(d):
               cov_a[ii,jj] = r**np.abs(ii-jj)
    elif cov_a_str == 'equi':
        mean_a = np.zeros(d)
        r = 0.2
        cov_a = r * np.ones((d,d)) + (1-r) * np.eye(d)

    Asy_cov = np.eye(d)  # asymptotic covariance matrix

    # SGD origial loop
    # set random seed for original samples
    results = Parallel(n_jobs=32)(delayed(main_loop)(seed, x_star, x_0, n, R, eta, var_epsilon, mean_a, cov_a, alpha, num_trials) for seed in range(1, 1+num_trials))
    mean_len_history = []
    std_len_history = []
    len_history = []
    cov_history = []
    x_out_history = []
    for ii in range(num_trials):
        mean_len_history.append(results[ii][0])
        std_len_history.append(results[ii][1])
        cov_history.append(results[ii][2])
        len_history.append(results[ii][3])
        x_out_history.append(results[ii][4])


    for seed in range(1, 1 + num_trials):
        # debug code
        print('*' * 20)
        print(f'Len: {mean_len_history[seed - 1]:.6f} ({std_len_history[seed - 1]:.10f})')
    print(np.mean(cov_history))
    # import pdb; pdb.set_trace()

    f = open(f'Result_{d}_{cov_a_str}.txt', 'a')
    f.write('----->\n')
    f.write(
        f'\t Cov Rate: {np.mean(cov_history)} \t ({np.std(cov_history)}) \tAvg Len: {np.mean(len_history)} \t ({np.std(len_history)/num_trials}) \n')
    f.write(f'\t d: {d} \t n: {n} \t R: {R} \t eta_0: {eta} \t alpha: {alpha} \t # Trials: {num_trials}\n')
    f.write(f'\t True solution:           [')
    for ii in range(d):
        f.write(f'{x_star[ii]:.6f}, ')
    f.write(']\n')
    f.write(f'\t center in last trial:    [')
    for ii in range(d):
        f.write(f'{x_out_history[-1][ii]:.6f}, ')
    f.write(']\n')
    f.write(f'\t CI UB in the last trial: [')
    for ii in range(d):
        f.write(f'{len_history[-1][ii] + x_out_history[-1][ii]:.6f}, ')
    f.write(']\n')
    f.write(f'\t CI LB in the last trial: [')
    for ii in range(d):
        f.write(f'{-len_history[-1][ii] + x_out_history[-1][ii]:.6f}, ')
    f.write(']\n')
    # f.write(f'\t Cover in the last trial: [')
    # for ii in range(d):
    #     f.write(f'{(cov_history)[-1][ii]:.0f}       , ')
    # f.write(']\n')

    f.close()

    return

def main_experiments_parallel_std(d, n, eta, alpha, x_star, x_0, R, var_epsilon, cov_a_str, num_trials):
    # mean and variance for generating a_i
    # identity covariance matrix case
    #
    # linear regression model:
    # b_i = x_star^\top a_i + \epsilon_i
    if cov_a_str == 'identity':
        mean_a = np.zeros(d)
        cov_a = np.eye(d)
    elif cov_a_str == 'toeplitz':
        mean_a = np.zeros(d)
        cov_a = np.eye(d)
        r = 0.5
        for ii in range(d):
            for jj in range(d):
                cov_a[ii, jj] = r ** np.abs(ii - jj)
    elif cov_a_str == 'equi':
        mean_a = np.zeros(d)
        r = 0.2
        cov_a = r * np.ones((d, d)) + (1 - r) * np.eye(d)

    # SGD origial loop
    # set random seed for original samples
    results = Parallel(n_jobs=32)(delayed(main_loop_std)(seed, x_star, x_0, n, R, eta, var_epsilon, mean_a, cov_a, alpha, num_trials) for seed in range(1, 1+num_trials))
    mean_len_history = []
    std_len_history = []
    len_history = []
    cov_history = []
    x_out_history = []
    for ii in range(num_trials):
        mean_len_history.append(results[ii][0])
        std_len_history.append(results[ii][1])
        cov_history.append(results[ii][2])
        len_history.append(results[ii][3])
        x_out_history.append(results[ii][4])


    for seed in range(1, 1 + num_trials):
        # debug code
        print('*' * 20)
        print(f'Len: {mean_len_history[seed - 1]:.6f} ({std_len_history[seed - 1]:.10f})')
    print(np.mean(cov_history))
    # import pdb; pdb.set_trace()

    f = open(f'Result_std_{d}_{cov_a_str}.txt', 'a')
    f.write('----->\n')
    f.write(
        f'\t Cov Rate: {np.mean(cov_history)} \t ({np.std(cov_history)}) \tAvg Len: {np.mean(len_history)} \t ({np.std(len_history)/num_trials}) \n')
    f.write(f'\t d: {d} \t n: {n} \t R: {R} \t eta_0: {eta} \t alpha: {alpha} \t # Trials: {num_trials}\n')
    f.write(f'\t True solution:           [')
    for ii in range(d):
        f.write(f'{x_star[ii]:.6f}, ')
    f.write(']\n')
    f.write(f'\t center in last trial:    [')
    for ii in range(d):
        f.write(f'{x_out_history[-1][ii]:.6f}, ')
    f.write(']\n')
    f.write(f'\t CI UB in the last trial: [')
    for ii in range(d):
        f.write(f'{len_history[-1][ii] + x_out_history[-1][ii]:.6f}, ')
    f.write(']\n')
    f.write(f'\t CI LB in the last trial: [')
    for ii in range(d):
        f.write(f'{-len_history[-1][ii] + x_out_history[-1][ii]:.6f}, ')
    f.write(']\n')
    # f.write(f'\t Cover in the last trial: [')
    # for ii in range(d):
    #     f.write(f'{(cov_history)[-1][ii]:.0f}       , ')
    # f.write(']\n')

    f.close()

    return

def main_experiments_parallel_plug_in(d, n, eta, alpha, x_star, x_0, var_epsilon, cov_a_str, num_trials):
    # mean and variance for generating a_i
    # identity covariance matrix case
    #
    # linear regression model:
    # b_i = x_star^\top a_i + \epsilon_i
    if cov_a_str == 'identity':
        mean_a = np.zeros(d)
        cov_a = np.eye(d)
    elif cov_a_str == 'toeplitz':
        mean_a = np.zeros(d)
        cov_a = np.eye(d)
        r = 0.5
        for ii in range(d):
            for jj in range(d):
                cov_a[ii, jj] = r ** np.abs(ii - jj)
    elif cov_a_str == 'equi':
        mean_a = np.zeros(d)
        r = 0.2
        cov_a = r * np.ones((d, d)) + (1 - r) * np.eye(d)

    # SGD origial loop
    # set random seed for original samples
    results = Parallel(n_jobs=32)(delayed(main_loop_plug_in)(seed, x_star, x_0, n, eta, var_epsilon, mean_a, cov_a, alpha, num_trials) for seed in range(1, 1+num_trials))
    # main_loop_plug_in(1, x_star, x_0, n, eta, var_epsilon, alpha, num_trials)
    mean_len_history = []
    std_len_history = []
    len_history = []
    cov_history = []
    x_out_history = []
    for ii in range(num_trials):
        mean_len_history.append(results[ii][0])
        std_len_history.append(results[ii][1])
        cov_history.append(results[ii][2])
        len_history.append(results[ii][3])
        x_out_history.append(results[ii][4])


    for seed in range(1, 1 + num_trials):
        # debug code
        print('*' * 20)
        print(f'Len: {mean_len_history[seed - 1]:.6f} ({std_len_history[seed - 1]:.10f})')
    print(np.mean(cov_history))
    # import pdb; pdb.set_trace()

    f = open(f'Result_PI_{d}_{cov_a_str}.txt', 'a')
    f.write('----->\n')
    f.write(
        f'\t Cov Rate: {np.mean(cov_history)} \t ({np.std(cov_history)}) \tAvg Len: {np.mean(len_history)} \t ({np.std(len_history)/num_trials}) \n')
    f.write(f'\t d: {d} \t n: {n} \t R: N.A. \t eta_0: {eta} \t alpha: {alpha} \t # Trials: {num_trials}\n')
    f.write(f'\t True solution:           [')
    for ii in range(d):
        f.write(f'{x_star[ii]:.6f}, ')
    f.write(']\n')
    f.write(f'\t center in last trial:    [')
    for ii in range(d):
        f.write(f'{x_out_history[-1][ii]:.6f}, ')
    f.write(']\n')
    f.write(f'\t CI UB in the last trial: [')
    for ii in range(d):
        f.write(f'{len_history[-1][ii] + x_out_history[-1][ii]:.6f}, ')
    f.write(']\n')
    f.write(f'\t CI LB in the last trial: [')
    for ii in range(d):
        f.write(f'{-len_history[-1][ii] + x_out_history[-1][ii]:.6f}, ')
    f.write(']\n')
    # f.write(f'\t Cover in the last trial: [')
    # for ii in range(d):
    #     f.write(f'{(cov_history)[-1][ii]:.0f}       , ')
    # f.write(']\n')

    f.close()

    return

def main_experiments_parallel_BM(d, n, eta, alpha, x_star, x_0, M_ratio, var_epsilon, cov_a_str, num_trials):
    # mean and variance for generating a_i
    # identity covariance matrix case
    #
    # linear regression model:
    # b_i = x_star^\top a_i + \epsilon_i
    if cov_a_str == 'identity':
        mean_a = np.zeros(d)
        cov_a = np.eye(d)
    elif cov_a_str == 'toeplitz':
        mean_a = np.zeros(d)
        cov_a = np.eye(d)
        r = 0.5
        for ii in range(d):
            for jj in range(d):
                cov_a[ii, jj] = r ** np.abs(ii - jj)
    elif cov_a_str == 'equi':
        mean_a = np.zeros(d)
        r = 0.2
        cov_a = r * np.ones((d, d)) + (1 - r) * np.eye(d)

    # SGD origial loop
    # set random seed for original samples
    M = int(np.floor(n ** (M_ratio)))-1
    N = int(np.floor(n**(1-alpha)/(M+1)))
    results = Parallel(n_jobs=32)(delayed(main_loop_BM)(seed, x_star, x_0, M, N, n, eta, var_epsilon, mean_a, cov_a, alpha, num_trials) for seed in range(1, 1+num_trials))
    # main_loop_BM(1, x_star, x_0, M, N, n, eta, var_epsilon, alpha, num_trials)
    mean_len_history = []
    std_len_history = []
    len_history = []
    cov_history = []
    x_out_history = []
    for ii in range(num_trials):
        mean_len_history.append(results[ii][0])
        std_len_history.append(results[ii][1])
        cov_history.append(results[ii][2])
        len_history.append(results[ii][3])
        x_out_history.append(results[ii][4])


    for seed in range(1, 1 + num_trials):
        # debug code
        print('*' * 20)
        print(f'Len: {mean_len_history[seed - 1]:.6f} ({std_len_history[seed - 1]:.10f})')
    print(np.mean(cov_history))
    # import pdb; pdb.set_trace()

    f = open(f'Result_BM_{d}_{cov_a_str}.txt', 'a')
    f.write('----->\n')
    f.write(
        f'\t Cov Rate: {np.mean(cov_history)} \t ({np.std(cov_history)}) \tAvg Len: {np.mean(len_history)} \t ({np.std(len_history)/num_trials}) \n')
    f.write(f'\t d: {d} \t n: {n} \t M ratio: {M_ratio} \t eta_0: {eta} \t alpha: {alpha} \t # Trials: {num_trials}\n')
    f.write(f'\t True solution:           [')
    for ii in range(d):
        f.write(f'{x_star[ii]:.6f}, ')
    f.write(']\n')
    f.write(f'\t center in last trial:    [')
    for ii in range(d):
        f.write(f'{x_out_history[-1][ii]:.6f}, ')
    f.write(']\n')
    f.write(f'\t CI UB in the last trial: [')
    for ii in range(d):
        f.write(f'{len_history[-1][ii] + x_out_history[-1][ii]:.6f}, ')
    f.write(']\n')
    f.write(f'\t CI LB in the last trial: [')
    for ii in range(d):
        f.write(f'{-len_history[-1][ii] + x_out_history[-1][ii]:.6f}, ')
    f.write(']\n')
    # f.write(f'\t Cover in the last trial: [')
    # for ii in range(d):
    #     f.write(f'{(cov_history)[-1][ii]:.0f}       , ')
    # f.write(']\n')

    f.close()

    return

def main_experiments_parallel_BM_v2(d, n, eta, alpha, x_star, x_0, M_ratio, var_epsilon, cov_a_str, num_trials):
    # mean and variance for generating a_i
    # identity covariance matrix case
    #
    # linear regression model:
    # b_i = x_star^\top a_i + \epsilon_i
    if cov_a_str == 'identity':
        mean_a = np.zeros(d)
        cov_a = np.eye(d)
    elif cov_a_str == 'toeplitz':
        mean_a = np.zeros(d)
        cov_a = np.eye(d)
        r = 0.5
        for ii in range(d):
            for jj in range(d):
                cov_a[ii, jj] = r ** np.abs(ii - jj)
    elif cov_a_str == 'equi':
        mean_a = np.zeros(d)
        r = 0.2
        cov_a = r * np.ones((d, d)) + (1 - r) * np.eye(d)

    # SGD origial loop
    # set random seed for original samples
    M = int(np.floor(n ** (M_ratio)))-1
    N = int(np.floor(n**(1-alpha)/(M+1)))
    results = Parallel(n_jobs=32)(delayed(main_loop_BM)(seed, x_star, x_0, M, N, n, eta, var_epsilon, mean_a, cov_a, alpha, num_trials) for seed in range(1, 1+num_trials))
    # main_loop_BM(1, x_star, x_0, M, N, n, eta, var_epsilon, alpha, num_trials)
    mean_len_history = []
    std_len_history = []
    len_history = []
    cov_history = []
    x_out_history = []
    for ii in range(num_trials):
        mean_len_history.append(results[ii][0])
        std_len_history.append(results[ii][1])
        cov_history.append(results[ii][2])
        len_history.append(results[ii][3])
        x_out_history.append(results[ii][4])


    for seed in range(1, 1 + num_trials):
        # debug code
        print('*' * 20)
        print(f'Len: {mean_len_history[seed - 1]:.6f} ({std_len_history[seed - 1]:.10f})')
    print(np.mean(cov_history))
    # import pdb; pdb.set_trace()

    f = open(f'Result_BM_WS_{d}_{cov_a_str}.txt', 'a')
    f.write('----->\n')
    f.write(
        f'\t Cov Rate: {np.mean(cov_history)} \t ({np.std(cov_history)}) \tAvg Len: {np.mean(len_history)} \t ({np.std(len_history)/num_trials}) \n')
    f.write(f'\t d: {d} \t n: {n} \t M ratio: {M_ratio} \t eta_0: {eta} \t alpha: {alpha} \t # Trials: {num_trials}\n')
    f.write(f'\t True solution:           [')
    for ii in range(d):
        f.write(f'{x_star[ii]:.6f}, ')
    f.write(']\n')
    f.write(f'\t center in last trial:    [')
    for ii in range(d):
        f.write(f'{x_out_history[-1][ii]:.6f}, ')
    f.write(']\n')
    f.write(f'\t CI UB in the last trial: [')
    for ii in range(d):
        f.write(f'{len_history[-1][ii] + x_out_history[-1][ii]:.6f}, ')
    f.write(']\n')
    f.write(f'\t CI LB in the last trial: [')
    for ii in range(d):
        f.write(f'{-len_history[-1][ii] + x_out_history[-1][ii]:.6f}, ')
    f.write(']\n')
    # f.write(f'\t Cover in the last trial: [')
    # for ii in range(d):
    #     f.write(f'{(cov_history)[-1][ii]:.0f}       , ')
    # f.write(']\n')

    f.close()

    return

def main_experiments_parallel_wo(d, n, eta, alpha, x_star, x_0, R, var_epsilon, cov_a_str, num_trials):
    # mean and variance for generating a_i
    # identity covariance matrix case
    #
    # linear regression model:
    # b_i = x_star^\top a_i + \epsilon_i
    if cov_a_str == 'identity':
        mean_a = np.zeros(d)
        cov_a = np.eye(d)
    elif cov_a_str == 'toeplitz':
        mean_a = np.zeros(d)
        cov_a = np.eye(d)
        r = 0.5
        for ii in range(d):
            for jj in range(d):
                cov_a[ii, jj] = r ** np.abs(ii - jj)
    elif cov_a_str == 'equi':
        mean_a = np.zeros(d)
        r = 0.2
        cov_a = r * np.ones((d, d)) + (1 - r) * np.eye(d)

    # SGD origial loop
    # set random seed for original samples
    results = Parallel(n_jobs=32)(delayed(main_loop_wo)(seed, x_star, x_0, n, R, eta, var_epsilon, mean_a, cov_a, alpha, num_trials) for seed in range(1, 1+num_trials))
    mean_len_history = []
    std_len_history = []
    len_history = []
    cov_history = []
    x_out_history = []
    for ii in range(num_trials):
        mean_len_history.append(results[ii][0])
        std_len_history.append(results[ii][1])
        cov_history.append(results[ii][2])
        len_history.append(results[ii][3])
        x_out_history.append(results[ii][4])


    for seed in range(1, 1 + num_trials):
        # debug code
        print('*' * 20)
        print(f'Len: {mean_len_history[seed - 1]:.6f} ({std_len_history[seed - 1]:.10f})')
    print(np.mean(cov_history))
    # import pdb; pdb.set_trace()

    f = open(f'Result_wo_{d}_{cov_a_str}.txt', 'a')
    f.write('----->\n')
    f.write(
        f'\t Cov Rate: {np.mean(cov_history)} \t ({np.std(cov_history)}) \tAvg Len: {np.mean(len_history)} \t ({np.std(len_history)/num_trials}) \n')
    f.write(f'\t d: {d} \t n: {n} \t R: {R} \t eta_0: {eta} \t alpha: {alpha} \t # Trials: {num_trials}\n')
    f.write(f'\t True solution:           [')
    for ii in range(d):
        f.write(f'{x_star[ii]:.6f}, ')
    f.write(']\n')
    f.write(f'\t center in last trial:    [')
    for ii in range(d):
        f.write(f'{x_out_history[-1][ii]:.6f}, ')
    f.write(']\n')
    f.write(f'\t CI UB in the last trial: [')
    for ii in range(d):
        f.write(f'{len_history[-1][ii] + x_out_history[-1][ii]:.6f}, ')
    f.write(']\n')
    f.write(f'\t CI LB in the last trial: [')
    for ii in range(d):
        f.write(f'{-len_history[-1][ii] + x_out_history[-1][ii]:.6f}, ')
    f.write(']\n')
    # f.write(f'\t Cover in the last trial: [')
    # for ii in range(d):
    #     f.write(f'{(cov_history)[-1][ii]:.0f}       , ')
    # f.write(']\n')

    f.close()

    return

if __name__ == '__main__':
    # basic setting
    var_epsilon = 1  # variance for noise in linear regression
    d = 1  # d = 5,20,100,200
    n = int(1e5)  # sample size
    eta = 1e-2
    alpha = 0.501  # step size eta_i = eta * i^{-alpha}
    x_star = np.linspace(0, 1, d)  # optimal solution
    x_0 = np.zeros(d)  # initial guess
    R = 2  # number of bootstrap
    num_trials = 500

    for R in [2,5,10]:
        # main_experiments(d, n, eta, alpha, x_star, x_0, R, var_epsilon, num_trials)
        main_experiments_parallel(d, n, eta, alpha, x_star, x_0, R, var_epsilon, num_trials)


