import numpy as np
import json
import argparse
import logging
import torch
import time
import gc
from pathlib import Path
from itertools import product

import config
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


def convert_tensor_scalar(value):
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return value.item()
    
    return value

def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def run_sweep(param_grid, jsonl_file_path=None):
    if jsonl_file_path is None:
        jsonl_file_path = RESULTS_DIR / f'experiment_results_{int(time.time())}.jsonl'

    keys, values = zip(*param_grid.items())
    for combo in product(*values):
        params = dict(zip(keys, combo))
        logger.info(f'Running experiment with parameters: {params}')
        run_dict = run_experiment(params)

        k = params['k']
        d = params['d']
        N_ = params['N']

        kappa = k / d
        tau = N_ / (d * d)

        jsonl_entry = {
            'beta': params['beta'],
            'alpha': params['alpha'],
            'rho': params['rho'],
            'strength': params['strengths'],
            'kappa': kappa,
            'tau': tau,
            'cov_type': 'identity',
            'mse_runs': run_dict['mse'],
            'mean_mse': np.mean(run_dict['mse']),
            'std_mse': np.std(run_dict['mse']),
            'mean_e_TF': np.mean(run_dict['e_TF']),
            'std_e_TF': np.std(run_dict['e_TF']),
            'mean_e_ICL': np.mean(run_dict['e_ICL']),
            'std_e_ICL': np.std(run_dict['e_ICL']),
            'mean_e_IDG': np.mean(run_dict['e_IDG']),
            'std_e_IDG': np.std(run_dict['e_IDG']),
            'mean_diff': np.mean(run_dict['diff']),
            'std_diff': np.std(run_dict['diff'])
        }
        with open(jsonl_file_path, 'a') as f:
            f.write(json.dumps(jsonl_entry) + '\n')

        cleanup_cuda()

def run_experiment(params):
    lam = 1e-9
    monte_carlo_runs = 10
    

    N = params['N']
    N_test = 100
    alpha = params['alpha']
    beta = params['beta']
    rho = params['rho']
    strength = params['strengths']
    k = params['k']
    d = params['d']

    kappa = k / d
    tau = N / (d * d)

    run_dict = {
        'mse': [],
        'e_TF': [],
        'e_ICL': [],
        'e_IDG': [],
        'diff': []
    }

    ell = int(alpha * d)
    for run in range(monte_carlo_runs):
        logger.info(f'Experiment starting: beta={beta}, alpha={alpha}, ell={ell}, d={d}, rho={rho}, kappa={kappa}, tau={tau}, strength={strength}, cov_type=identity, run={run}')
        
        with torch.inference_mode():
            y_train, x_train, w_train, w_task_family_train, w_cov, epsilon = draw_pretraining_torch(
                N, d, k, ell, rho, strength, cov_type='identity', device=config.DEVICE
            )
            Gamma = gamma_star_torch(y_train, x_train, lam)
            mse = test_error_torch(Gamma, N_test, beta, ell, rho, w_task_family_train, d)

        try:
            e_ICL = e_(Gamma, alpha, rho, w_task_family_train=w_task_family_train, error_type='ICL')
            e_IDG = e_(Gamma, alpha, rho, w_task_family_train=w_task_family_train, error_type='IDG')
            diff = diff_matrix(Gamma, rho, beta, w_task_family_train)

            e_TF_ = e_TF(beta, e_ICL, e_IDG, diff)
            e_TF_, e_ICL, e_IDG, diff = map(convert_tensor_scalar, [e_TF_, e_ICL, e_IDG, diff])

        except Exception as e:
            logger.error(f'Error computing e_TF: {e}')

        run_dict['mse'].append(mse)
        run_dict['e_TF'].append(e_TF_)
        run_dict['e_ICL'].append(e_ICL)
        run_dict['e_IDG'].append(e_IDG)
        run_dict['diff'].append(diff)
        logger.info(f"""beta={beta}, alpha={alpha}, ell={ell}, d={d}, rho={rho}, strength={strength},
                        cov_type=identity, run={run}, mse={mse:.4f}, e_TF={e_TF_:.4f}, e_ICL={e_ICL:.4f}, e_IDG={e_IDG:.4f}, diff={diff:.4f}""")

    logger.info(f'End iteration with alpha: {alpha}, beta: {beta}, rho: {rho}, kappa: {kappa}, tau: {tau}, strength: {strength}')

    return run_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ICL task familiarity experiments')
    # parser.add_argument('--jsonl_file_path', type=str, default='experiment_results.jsonl', help='Path to save experiment results in JSONL format')
    parser.add_argument('--task_id', type=int, default=0, help='SLURM task id')
    args = parser.parse_args()

    d = 100

    betas = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    # betas = betas[::-1]
    alphas = [2.35, 5.5, 9.62, 22.7, 100]
    rhos = [0.1, 0.5, 1.0, 2.0]
    kappas = [0.1, 0.25, 0.5, 0.75, 1.0, 2.0]
    taus = [0.1, 0.25, 0.5, 1.0]
    # strengths = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    strengths = [0.0]


    param_grid = {
        'd': [d],
        'beta': betas,
        'alpha': alphas,
        'rho': rhos,
        'strengths': strengths,
        'N': [int(tau * d * d) for tau in taus],
        'k': [int(kappa * d) for kappa in kappas]
    }

    run_sweep(param_grid)

