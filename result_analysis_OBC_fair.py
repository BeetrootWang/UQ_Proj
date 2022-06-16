# Process the result of numerical experiments
import numpy as np
cnt = 0
for B in [3,5,10]:
    result_str = f'rerun ($B={B}$) '
    for target_d in [5,20,200]:
        target_n = int(1e4)
        best_cov = 0 # Corresponds to R = 3,5,10,50
        best_cov_var = 0
        best_len = 1e5
        best_len_var = 0
        filename = f'Result_simplererun{B}_{target_d}_equi.txt'
        # print(filename)
        f = open(filename, 'r')
        R_or_Ratio = 'something wrong'
        for line in f:
            if '-->' in line:
                cnt = 1
                if R_or_Ratio!='something wrong' and abs(cov-0.95) + Len < abs(best_cov - 0.95) + best_len:
                    best_cov = cov
                    best_cov_var = cov_var / np.sqrt(target_n)
                    best_len = Len
                    best_len_var = Len_var
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
                if 'BM' in filename:
                    if n!= int(target_n / B):
                        continue
                    R_or_Ratio = tmp[5+1]
                    eta_0 = float(tmp[7+1])
                    alpha = float(tmp[9+1])
                    num_trials = int(tmp[12+1])
                else:
                    if n!= int(target_n / B):
                        cov=0
                        Len=1e5
                        continue
                    R_or_Ratio = tmp[5]
                    eta_0 = float(tmp[7])
                    alpha = float(tmp[9])
                    num_trials = int(tmp[12])
        result_str = result_str + f'& ${best_cov*100:.2f}$ (${best_cov_var*100:.2f}$) & ${best_len*100:.2f}$ (${best_len_var*100:.2f}$)'
        # import pdb; pdb.set_trace()
        f.close()

    result_str = result_str+' \\\\ \hline'
    print(result_str)