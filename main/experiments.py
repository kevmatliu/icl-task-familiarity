import numpy as np
import json
import argparse
import logging
import torch
import time
from pathlib import Path

from utils import *

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = BASE_DIR / 'logs_stdout'
LOG_DIR.mkdir(parents=True, exist_ok=True)

log_file = LOG_DIR / f'run_{int(time.time())}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def run_experiments(jsonl_file_path='experiment_results.jsonl'):
    d = 100
    rho = 0.5
    betas = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    # betas = betas[::-1]
    alphas = [2.35, 5.5, 9.62, 22.7, 100, 1_000]

    lam = 1e-9
    monte_carlo_runs = 10
    
    # defaults
    N = 100
    k = 50
    N_test = 50
    seed = 10

    res = {}

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'

    for beta in betas:
        res[beta] = {}
        for alpha in alphas:
            mse_runs = []
            e_icl_trace_runs = []
            ell = int(alpha * d)
            for run in range(monte_carlo_runs):
                logger.info(f'Experiment starting: beta={beta}, alpha={alpha}, ell={ell}, d={d}, rho={rho}')
                
                y_train, x_train, w_train, w_task_family_train, epsilon = draw_pretraining_torch(N, d, k, ell, rho, seed + run, device=device)
                Gamma = gamma_star_torch(y_train, x_train, lam)
                mse = test_error_torch(Gamma, N_test, beta, ell, rho, w_task_family_train, d, seed)

                e_icl_trace_ = e_ICL_trace(Gamma, d, ell, rho, beta)
                
                mse_runs.append(mse)
                e_icl_trace_runs.append(e_icl_trace_)
                logger.info(f'   beta={beta}, alpha={alpha}, ell={ell}, d={d}, rho={rho}, run: {run}/{monte_carlo_runs}, run_mse: {mse}, e_icl_trace: {e_icl_trace_}')
            logger.info(f'End experiment with alpha: {alpha}, beta: {beta} | mse_runs: {mse_runs}, mean_mse: {np.mean(mse_runs)}, std_mse: {np.std(mse_runs)}')
            res[beta][alpha] = mse_runs

            jsonl_entry = {
                'beta': beta,
                'alpha': alpha,
                'mse_runs': mse_runs,
                'mean_mse': np.mean(mse_runs),
                'std_mse': np.std(mse_runs),
                'mean_e_ICL_trace': np.mean(e_icl_trace_runs),
                'std_e_ICL_trace': np.std(e_icl_trace_runs)
            }
            with open(jsonl_file_path, 'a') as f:
                f.write(json.dumps(jsonl_entry) + '\n')

    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ICL task familiarity experiments')
    # parser.add_argument('--jsonl_file_path', type=str, default='experiment_results.jsonl', help='Path to save experiment results in JSONL format')
    parser.add_argument('--task_id', type=int, default=0, help='SLURM task id')
    args = parser.parse_args()

    file_path = RESULTS_DIR / f'experiment_results_{int(time.time())}.jsonl'

    final_results = run_experiments(jsonl_file_path=file_path)
    with open('experiment_results_final.json', 'w') as f:
        json.dump(final_results, f, indent=4)


