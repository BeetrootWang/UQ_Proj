# Process the result of numerical experiments
import numpy as np
cnt = 0
result_str_3 = '$R=3$  '
result_str_5 = '$R=5$  '
result_str_10 = '$R=10$ '
for target_d in [5,20,200]:
    target_n = int(1e5)
    best_cov = [0,0,0,0] # Corresponds to R = 3,5,10,50
    best_cov_var = [0,0,0,0]
    best_len = [1e5, 1e5, 1e5, 1e5]
    best_len_var = [0,0,0,0]
    filename = f'Result_{target_d}_identity.txt'
    f = open(filename, 'r')
    R_or_Ratio = np.NaN
    ii=1
    for line in f:
        if '-->' in line:
            cnt = 1
            if R_or_Ratio == '3':
                ii=0
            elif R_or_Ratio == '5':
                ii=1
            elif R_or_Ratio == '10':
                # import pdb; pdb.set_trace()
                ii=2
            elif R_or_Ratio == '50':
                ii=3
            else:
                continue
        elif cnt == 1:
            cnt = 2
            # This line contains
            # Coverage rate and average length
            tmp = line.split(" ")
            cov = float(tmp[3])
            cov_var = float(tmp[5].strip('()'))
            Len = float(tmp[8])
            Len_var = float(tmp[10].strip('()'))
        elif cnt == 2:
            cnt = 3
            tmp = line.split()
            d = int(tmp[1])
            n = int(tmp[3])
            if n!= target_n:
                continue
            R_or_Ratio = tmp[5]
            eta_0 = float(tmp[7])
            alpha = float(tmp[9])
            num_trials = int(tmp[12])
        if abs(cov-0.95) + Len < abs(best_cov[ii] - 0.95) + best_len[ii]:
            best_cov[ii] = cov
            best_cov_var[ii] = cov_var / np.sqrt(target_n)
            best_len[ii] = Len
            best_len_var[ii] = Len_var
    result_str_3 = result_str_3 + f'& ${best_cov[0]*100:.2f}$ (${best_cov_var[0]*100:.2f}$) & ${best_len[0]*100:.2f}$ (${best_len_var[0]*100:.2f}$)'
    result_str_5 = result_str_5 + f'& ${best_cov[1]*100:.2f}$ (${best_cov_var[1]*100:.2f}$) & ${best_len[1]*100:.2f}$ (${best_len_var[1]*100:.2f}$)'
    result_str_10 = result_str_10 + f'& ${best_cov[2]*100:.2f}$ (${best_cov_var[2]*100:.2f}$) & ${best_len[2]*100:.2f}$ (${best_len_var[2]*100:.2f}$)'
    # import pdb; pdb.set_trace()
    f.close()

result_str_3 = result_str_3+'\\\\ \hline'
result_str_5 = result_str_5+'\\\\ \hline'
result_str_10 = result_str_10+'\\\\ \hline'
print(result_str_3)
print(result_str_5)
print(result_str_10)