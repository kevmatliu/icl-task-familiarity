import torch
import config

from .tensor import convert_tensor_scalar
from .error_metrics import e_, diff_matrix, e_TF
from .batches_train import gamma_star_torch_streaming, gamma_star_torch_streaming_batched
from .batches_test import test_error_torch_streaming, test_error_torch_streaming_batched


def run_single_experiment_streaming(
    *,
    alpha,
    beta,
    d,
    rho,
    kappa,
    tau,
    strength=0.0,
    cov_type='identity',
    lam=None,
    batch_size_train=64,
    batch_size_test=256,
    device=config.DEVICE,
    sample_dtype=torch.float32,
    solve_dtype=torch.float32,
):
    """
    End-to-end experiment
    """
    N = int(tau * d * d)
    ell = int(alpha * d)
    k = int(kappa * d)

    if lam is None:
        lam = 0.0

    Gamma, w_task_family_train, w_cov = gamma_star_torch_streaming(
        N=N,
        d=d,
        k=k,
        ell=ell,
        rho=rho,
        lam=lam,
        strength=strength,
        cov_type=cov_type,
        batch_size=batch_size_train,
        device=device,
        sample_dtype=sample_dtype,
        solve_dtype=solve_dtype,
    )

    mse = test_error_torch_streaming(
        Gamma=Gamma,
        N_test=N,
        beta=beta,
        ell=ell,
        rho=rho,
        w_task_family_train=w_task_family_train,
        d=d,
        batch_size=batch_size_test,
        device=device,
        sample_dtype=sample_dtype,
    )

    Gamma_th = Gamma.to(dtype=solve_dtype)

    e_ICL = convert_tensor_scalar(e_(Gamma_th, alpha, rho, w_task_family_train, error_type='ICL'))
    e_IDG = convert_tensor_scalar(e_(Gamma_th, alpha, rho, w_task_family_train, error_type='IDG'))
    diff = convert_tensor_scalar(diff_matrix(Gamma_th, rho, beta, w_task_family_train).item())
    tf = e_TF(beta, e_ICL, e_IDG, diff)

    return {
        'alpha': alpha,
        'beta': beta,
        'd': d,
        'ell': ell,
        'rho': rho,
        'kappa': kappa,
        'k': k,
        'tau': tau,
        'strength': strength,
        'cov_type': cov_type,
        'lam': lam,
        'mse': mse,
        'e_ICL': e_ICL,
        'e_IDG': e_IDG,
        'e_TF': tf,
        'diff': diff,
    }


def run_single_experiment_batched_mc(
    *,
    alpha,
    beta,
    d,
    rho,
    kappa,
    tau,
    monte_carlo_runs=10,
    strength=0.0,
    cov_type='identity',
    lam=None,
    batch_size_train=64,
    batch_size_test=256,
    device=config.DEVICE,
    sample_dtype=torch.float32,
    solve_dtype=torch.float32,
):
    """
    Vectorized end-to-end experiment for multiple Monte Carlo runs.

    Returns the same metric names as run_experiment expects, each as a list.
    """
    N = int(tau * d * d)
    ell = int(alpha * d)
    k = int(kappa * d)
    if lam is None:
        lam = 0.0

    Gamma, w_task_family_train, _ = gamma_star_torch_streaming_batched(
        num_runs=monte_carlo_runs,
        N=N,
        d=d,
        k=k,
        ell=ell,
        rho=rho,
        lam=lam,
        strength=strength,
        cov_type=cov_type,
        batch_size=batch_size_train,
        device=device,
        sample_dtype=sample_dtype,
        solve_dtype=solve_dtype,
    )

    mse = test_error_torch_streaming_batched(
        Gamma=Gamma,
        N_test=N,
        beta=beta,
        ell=ell,
        rho=rho,
        w_task_family_train=w_task_family_train,
        d=d,
        batch_size=batch_size_test,
        device=device,
        sample_dtype=sample_dtype,
    )

    e_ICL_vals = []
    e_IDG_vals = []
    diff_vals = []
    e_TF_vals = []
    Gamma_th = Gamma.to(dtype=solve_dtype)

    for r in range(monte_carlo_runs):
        w_r = w_task_family_train[r]
        Gamma_r = Gamma_th[r]
        e_ICL_r = convert_tensor_scalar(e_(Gamma_r, alpha, rho, w_r, error_type='ICL'))
        e_IDG_r = convert_tensor_scalar(e_(Gamma_r, alpha, rho, w_r, error_type='IDG'))
        diff_r = convert_tensor_scalar(diff_matrix(Gamma_r, rho, beta, w_r))
        tf_r = e_TF(beta, e_ICL_r, e_IDG_r, diff_r)

        e_ICL_vals.append(float(e_ICL_r))
        e_IDG_vals.append(float(e_IDG_r))
        diff_vals.append(float(diff_r))
        e_TF_vals.append(float(tf_r))

    return {
        'alpha': alpha,
        'beta': beta,
        'd': d,
        'ell': ell,
        'rho': rho,
        'kappa': kappa,
        'k': k,
        'tau': tau,
        'strength': strength,
        'cov_type': cov_type,
        'lam': lam,
        'mse': [float(x) for x in mse],
        'e_ICL': e_ICL_vals,
        'e_IDG': e_IDG_vals,
        'e_TF': e_TF_vals,
        'diff': diff_vals,
    }


def run_single_experiment(
    *,
    alpha,
    beta,
    d,
    rho,
    kappa,
    tau,
    strength=0.0,
    cov_type='identity',
    lam=1e-9,
    batch_size_train=64,
    batch_size_test=256,
    device=config.DEVICE,
    sample_dtype=torch.float32,
    solve_dtype=torch.float32,
):
    """
    Keeping legacy API name.
    """
    return run_single_experiment_streaming(
        alpha=alpha,
        beta=beta,
        d=d,
        rho=rho,
        kappa=kappa,
        tau=tau,
        strength=strength,
        cov_type=cov_type,
        lam=lam,
        batch_size_train=batch_size_train,
        batch_size_test=batch_size_test,
        device=device,
        sample_dtype=sample_dtype,
        solve_dtype=solve_dtype,
    )
