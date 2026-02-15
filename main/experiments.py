import numpy as np
import json
import argparse

from utils import *


def run_experiments(jsonl_file_path='experiment_results.jsonl'):
    d = 100
    rho = 0.1
    betas = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    betas = betas[::-1]
    alphas = [2.35, 5.5, 9.62, 22.7, 100, 1000]

    lam = 1e-9
    monte_carlo_runs = 10
    
    # defaults
    N = 100
    k = 50
    N_test = 50
    seed = 10

    res = {}

    for beta in betas:
        res[beta] = {}
        for alpha in alphas:
            mse_runs = []
            for run in range(monte_carlo_runs):
                ell = int(alpha * d)
                print(f'Experiment: beta={beta}, alpha={alpha}, ell={ell}, d={d}, rho={rho}')
                
                y_train, x_train, w_train, w_task_family_train, epsilon = draw_pretraining(N, d, k, ell, rho, seed + run)
                Gamma = gamma_star(y_train, x_train, lam)
                mse = test_error(Gamma, N_test, beta, ell, rho, w_task_family_train, d, seed)
                
                mse_runs.append(mse)
                print(f'   run: {run}/{monte_carlo_runs}, run_mse: {mse}')
            print(f'End experiment with alpha: {alpha}, beta: {beta} | mse_runs: {mse_runs}, mean_mse: {np.mean(mse_runs)}, std_mse: {np.std(mse_runs)}')
            print('-' * 50)
            res[beta][alpha] = mse_runs

            jsonl_entry = {
                'beta': beta,
                'alpha': alpha,
                'mse_runs': mse_runs,
                'mean_mse': np.mean(mse_runs),
                'std_mse': np.std(mse_runs),
            }
            with open(jsonl_file_path, 'a') as f:
                f.write(json.dumps(jsonl_entry) + '\n')

    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ICL task familiarity experiments')
    parser.add_argument('--jsonl_file_path', type=str, default='experiment_results.jsonl', help='Path to save experiment results in JSONL format')
    parser.add_argument('--task_id', type=int, default=0, help='SLURM task id')
    args = parser.parse_args()


    final_results = run_experiments()
    print('Final Results:', final_results)
    with open('experiment_results.json', 'w') as f:
        json.dump(final_results, f, indent=4)


