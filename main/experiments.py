import os
import json
import argparse
import logging
import time
import gc
from pathlib import Path
from itertools import product

import numpy as np
import torch

import config
try:
    from utils import run_single_experiment_batched_mc
except ImportError:
    from utils_cluster import run_single_experiment_batched_mc

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / 'results' / 'tau'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = BASE_DIR / 'logs_stdout'
LOG_DIR.mkdir(parents=True, exist_ok=True)


def _slurm_int(name, default=None):
    value = os.environ.get(name)
    if value is None or value == '':
        return default
    return int(value)


def configure_torch_threads(cpus_per_task=None):
    """Respect --cpus-per-task for CPU-side Torch/OpenMP work."""
    if cpus_per_task is None:
        cpus_per_task = _slurm_int('SLURM_CPUS_PER_TASK', 1)
    cpus_per_task = max(1, int(cpus_per_task))

    torch.set_num_threads(cpus_per_task)
    torch.set_num_interop_threads(max(1, min(4, cpus_per_task)))
    os.environ.setdefault('OMP_NUM_THREADS', str(cpus_per_task))
    os.environ.setdefault('MKL_NUM_THREADS', str(cpus_per_task))
    return cpus_per_task


def setup_logging(task_id=None):
    suffix = f'task_{task_id}' if task_id is not None else f'run_{int(time.time())}'
    log_file = LOG_DIR / f'{suffix}_{int(time.time())}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True,
    )
    return logging.getLogger(__name__)


logger = setup_logging()


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _default_jsonl_path(task_id=None):
    if task_id is None:
        return RESULTS_DIR / f'experiment_results_{int(time.time())}.jsonl'
    return RESULTS_DIR / f'experiment_results_task_{task_id:04d}.jsonl'


def shard_combinations(param_grid, task_id=0, total_tasks=1):
    """
    Split the Cartesian parameter grid across SLURM array tasks.

    Uses numpy.array_split so all combinations are covered even when the grid
    size is not divisible by total_tasks.
    """
    if total_tasks <= 0:
        raise ValueError(f'total_tasks must be positive, got {total_tasks}')
    if task_id < 0 or task_id >= total_tasks:
        raise ValueError(f'task_id must be in [0, {total_tasks - 1}], got {task_id}')

    keys, values = zip(*param_grid.items())
    all_combinations = list(product(*values))
    index_shards = np.array_split(np.arange(len(all_combinations)), total_tasks)
    my_indices = index_shards[task_id]
    return keys, all_combinations, my_indices


def run_sweep(param_grid, task_id=0, total_tasks=1, jsonl_file_path=None, monte_carlo_runs=5, mc_batch_size=1):
    if jsonl_file_path is None:
        jsonl_file_path = _default_jsonl_path(task_id if total_tasks > 1 else None)
    jsonl_file_path = Path(jsonl_file_path)
    jsonl_file_path.parent.mkdir(parents=True, exist_ok=True)

    keys, all_combinations, my_indices = shard_combinations(param_grid, task_id, total_tasks)
    logger.info(
        'Task %s/%s handling %s of %s parameter combinations; writing to %s',
        task_id,
        total_tasks,
        len(my_indices),
        len(all_combinations),
        jsonl_file_path,
    )

    for local_i, combo_idx in enumerate(my_indices):
        combo = all_combinations[int(combo_idx)]
        params = dict(zip(keys, combo))
        logger.info('Running shard item %s/%s, global combo %s: %s', local_i + 1, len(my_indices), combo_idx, params)
        run_dict = run_experiment(params, monte_carlo_runs=monte_carlo_runs, mc_batch_size=mc_batch_size)

        strength = params.get('strength', params.get('strengths', 0.0))
        cov_type = params.get('cov_type', 'identity')
        jsonl_entry = {
            'task_id': task_id,
            'total_tasks': total_tasks,
            'combo_index': int(combo_idx),
            'beta': _json_safe(params['beta']),
            'alpha': _json_safe(params['alpha']),
            'rho': _json_safe(params['rho']),
            'strength': _json_safe(strength),
            'kappa': _json_safe(params['kappa']),
            'tau': _json_safe(params['tau']),
            'cov_type': cov_type,
            'monte_carlo_runs': monte_carlo_runs,
            'mc_batch_size': mc_batch_size,
            'mse_runs': run_dict['mse'],
            'mean_mse': float(np.mean(run_dict['mse'])),
            'std_mse': float(np.std(run_dict['mse'])),
            'mean_e_TF': float(np.mean(run_dict['e_TF'])),
            'std_e_TF': float(np.std(run_dict['e_TF'])),
            'mean_e_ICL': float(np.mean(run_dict['e_ICL'])),
            'std_e_ICL': float(np.std(run_dict['e_ICL'])),
            'mean_e_IDG': float(np.mean(run_dict['e_IDG'])),
            'std_e_IDG': float(np.std(run_dict['e_IDG'])),
            'mean_diff': float(np.mean(run_dict['diff'])),
            'std_diff': float(np.std(run_dict['diff'])),
        }
        with open(jsonl_file_path, 'a') as f:
            f.write(json.dumps(jsonl_entry) + '\n')

        cleanup_cuda()


def _empty_run_dict():
    return {
        'mse': [],
        'e_TF': [],
        'e_ICL': [],
        'e_IDG': [],
        'diff': [],
    }


def _extend_run_dict(dst, src):
    for key in dst:
        dst[key].extend([float(x) for x in src[key]])


def run_experiment(params, monte_carlo_runs=5, mc_batch_size=1):
    alpha = params['alpha']
    beta = params['beta']
    d = params['d']
    rho = params['rho']
    kappa = params['kappa']
    tau = params['tau']
    strength = params.get('strength', params.get('strengths', 0.0))
    cov_type = params.get('cov_type', 'identity')
    lam = params.get('lam', 1e-9)

    logger.info(
        'MC start: beta=%s, alpha=%s, d=%s, rho=%s, kappa=%s, tau=%s, strength=%s, cov_type=%s, total_runs=%s, mc_batch_size=%s',
        beta,
        alpha,
        d,
        rho,
        kappa,
        tau,
        strength,
        cov_type,
        monte_carlo_runs,
        mc_batch_size,
    )

    run_dict = _empty_run_dict()
    completed = 0
    while completed < monte_carlo_runs:
        cur_mc = min(mc_batch_size, monte_carlo_runs - completed)
        logger.info(
            'Starting MC sub-batch: runs %s-%s of %s, cur_mc=%s',
            completed,
            completed + cur_mc - 1,
            monte_carlo_runs,
            cur_mc,
        )

        res = run_single_experiment_batched_mc(
            alpha=alpha,
            beta=beta,
            d=d,
            rho=rho,
            kappa=kappa,
            tau=tau,
            monte_carlo_runs=cur_mc,
            strength=strength,
            cov_type=cov_type,
            lam=lam,
            device=config.DEVICE,
        )

        _extend_run_dict(run_dict, res)

        for local_run, (mse, e_tf, e_icl, e_idg, diff) in enumerate(
            zip(res['mse'], res['e_TF'], res['e_ICL'], res['e_IDG'], res['diff'])
        ):
            global_run = completed + local_run
            logger.info(
                'beta=%s, alpha=%s, d=%s, rho=%s, strength=%s, kappa=%s, tau=%s, cov_type=%s, run=%s, mse=%.4f, e_TF=%.4f, e_ICL=%.4f, e_IDG=%.4f, diff=%.4f',
                beta,
                alpha,
                d,
                rho,
                strength,
                kappa,
                tau,
                cov_type,
                global_run,
                mse,
                e_tf,
                e_icl,
                e_idg,
                diff,
            )

        completed += cur_mc
        cleanup_cuda()

    logger.info(
        'End MC iteration: alpha=%s, beta=%s, rho=%s, kappa=%s, tau=%s, strength=%s, total_completed=%s',
        alpha,
        beta,
        rho,
        kappa,
        tau,
        strength,
        len(run_dict['mse']),
    )
    return run_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ICL task familiarity experiments')
    parser.add_argument('--task_id', type=int, default=None, help='SLURM array task id; defaults to SLURM_ARRAY_TASK_ID or 0')
    parser.add_argument('--total_tasks', type=int, default=None, help='SLURM array task count; defaults to SLURM_ARRAY_TASK_COUNT or 1')
    parser.add_argument('--jsonl_file_path', type=str, default=None, help='Optional explicit output path. Defaults to one file per array task.')
    parser.add_argument('--monte_carlo_runs', type=int, default=5, help='Total independent Monte Carlo runs per parameter setting')
    parser.add_argument('--mc_batch_size', type=int, default=1, help='How many Monte Carlo runs to vectorize at once. Use 1 or 2 for d=128 to avoid OOM.')
    parser.add_argument('--cpus_per_task', type=int, default=None, help='Defaults to SLURM_CPUS_PER_TASK')
    args = parser.parse_args()

    task_id = args.task_id
    if task_id is None:
        task_id = _slurm_int('SLURM_ARRAY_TASK_ID', 0)

    total_tasks = args.total_tasks
    if total_tasks is None:
        total_tasks = _slurm_int('SLURM_ARRAY_TASK_COUNT', 1)

    logger = setup_logging(task_id=task_id)
    cpus = configure_torch_threads(args.cpus_per_task)
    logger.info('Using task_id=%s, total_tasks=%s, torch threads=%s, device=%s', task_id, total_tasks, cpus, config.DEVICE)

    d = 64
    betas = np.linspace(0.0, 1.0, 10)
    # alphas = np.logspace(np.log10(2.35), np.log10(1_000), 50)
    # alphas = np.logspace(np.log10(2.35), np.log10(100), 20)
    alphas = np.logspace(np.log10(2.35), np.log(100), 10)
    # rhos = [0.5, 1.0, 2.0]
    rhos = [0.5]
    kappas = np.linspace(0.1, 2.0, 10)
    taus = np.linspace(0.1, 2.0, 10)
    strengths = [0.0]

    param_grid = {
        'd': [d],
        'beta': betas,
        'alpha': alphas,
        'rho': rhos,
        'strength': strengths,
        'kappa': kappas,
        'tau': taus,
    }

    run_sweep(
        param_grid,
        task_id=task_id,
        total_tasks=total_tasks,
        jsonl_file_path=args.jsonl_file_path,
        monte_carlo_runs=args.monte_carlo_runs,
        mc_batch_size=args.mc_batch_size,
    )
