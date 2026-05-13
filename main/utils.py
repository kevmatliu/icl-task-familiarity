import numpy as np
import torch
import correlation as corr
import config


def convert_tensor_scalar(value):
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return value.item()

    return value

def _empirical_moments(w_task_family_train: torch.Tensor):
    k, d = w_task_family_train.shape

    b_train = torch.mean(w_task_family_train, dim=0)
    R_train = (w_task_family_train.T @ w_task_family_train) / k

    return b_train, R_train


def e_(Gamma, alpha, rho, w_task_family_train, error_type):
    k, d = w_task_family_train.shape

    if error_type == 'ICL':
        b_train = torch.zeros(d, device=Gamma.device, dtype=Gamma.dtype)
        R_train = torch.eye(d, device=Gamma.device, dtype=Gamma.dtype)
    elif error_type == 'IDG':
        b_train, R_train = _empirical_moments(
            w_task_family_train.to(device=Gamma.device, dtype=Gamma.dtype)
        )
    else:
        raise ValueError(f"Unknown error_type={error_type!r}; expected 'ICL' or 'IDG'")

    ell = alpha * d
    device = Gamma.device
    dtype = Gamma.dtype

    corr1 = 1.0 + 1.0 / ell
    corr2 = 1.0 + 2.0 / ell

    upper_right = (corr2 * (1.0 + rho) * b_train).reshape(d, 1)
    lower_left = upper_right.T
    lower_right = torch.tensor(
        [[corr2 * (1.0 + rho) ** 2]], device=device, dtype=dtype
    )

    A_train = torch.cat(
        [R_train, ((1.0 + rho) * b_train).reshape(1, d)], dim=0
    )

    B_upper = torch.cat(
        [((1.0 + rho) / alpha) * torch.eye(d, device=device, dtype=dtype) + corr1 * R_train, upper_right],
        dim=1,
    )
    B_lower = torch.cat([lower_left, lower_right], dim=1)

    B_train = torch.cat([B_upper, B_lower], dim=0)
    second_order = 1. / d * torch.trace(Gamma @ B_train @ Gamma.T)
    first_order = 2. / d * torch.trace(Gamma @ A_train)

    return second_order - first_order + (1. + rho)


def diff_matrix(Gamma, rho, beta, w_task_family_train):
    d, _ = Gamma.shape
    device = Gamma.device
    dtype = Gamma.dtype

    b_train, _ = _empirical_moments(w_task_family_train.to(device=device, dtype=dtype))

    Gamma_1 = Gamma[:, :d]
    Gamma_2 = Gamma[:, d:]
    scalar = (b_train.T @ (Gamma_1.T - torch.eye(d, device=device, dtype=dtype)) @ Gamma_2)

    return 2. * (1. + rho) * (beta - beta ** 2) / d * scalar.squeeze()


def e_TF(beta, e_ICL, e_IDG, diff):
    return (1 - beta ** 2) * e_ICL + beta ** 2 * e_IDG + diff


def sample_w_task_family_train(d, k, strength=0.0, cov_type='identity', device=config.DEVICE, dtype=torch.float32,):
    factory_kwargs = dict(device=device, dtype=dtype)

    if cov_type == 'identity':
        cov_func = corr.identity_cov
    elif cov_type == 'exp':
        cov_func = corr.exp_cov
    elif cov_type == 'powerlaw':
        cov_func = corr.powerlaw_cov
    else:
        raise ValueError(f"Unknown cov_type={cov_type}")

    sqrt_d = torch.sqrt(torch.tensor(float(d), **factory_kwargs))

    w_cov = cov_func(d, strength, device=device, dtype=dtype)
    w_task_family_train = corr.sample_task_weights(w_cov, d, k)

    w_task_family_train = (
        w_task_family_train
        / (torch.linalg.norm(w_task_family_train, dim=1, keepdim=True) + 1e-12)
        * sqrt_d
    )

    return w_task_family_train, w_cov


def _sample_training_batch_summaries(
    batch_size,
    d,
    ell,
    rho,
    w_task_family_train,
    device,
    dtype,
):
    """
    Generate training batch x or y of shape (batch, ell+1, d), saving CUDA space.

    H_batch: (batch_size, d * (d + 1))
    y_last: (batch_size,)
    """
    factory_kwargs = dict(device=device, dtype=dtype)

    sqrt_d = torch.sqrt(torch.tensor(float(d), **factory_kwargs))
    sqrt_rho = torch.sqrt(torch.tensor(float(rho), **factory_kwargs))

    k = w_task_family_train.shape[0]
    w_train_samples = torch.randint(0, k, (batch_size,), device=device)
    w = w_task_family_train[w_train_samples]  # (batch, d)

    s = torch.zeros(batch_size, d, **factory_kwargs)    # (batch, d), sum_t y_t x_t
    q = torch.zeros(batch_size, 1, **factory_kwargs)    # (batch, 1), sum_t y_t^2

    for _ in range(ell):
        x_t = torch.randn(batch_size, d, **factory_kwargs) / sqrt_d
        eps_t = torch.randn(batch_size, **factory_kwargs) * sqrt_rho
        y_t = torch.sum(x_t * w, dim=1) + eps_t

        s += y_t.unsqueeze(1) * x_t
        q += (y_t ** 2).unsqueeze(1)

    x_last = torch.randn(batch_size, d, **factory_kwargs) / sqrt_d
    eps_last = torch.randn(batch_size, **factory_kwargs) * sqrt_rho
    y_last = torch.sum(x_last * w, dim=1) + eps_last

    c = (float(d) / float(ell)) * s             # (batch, d)
    q = q / float(ell)                          # (batch, 1)

    hz_block = torch.cat([c, q], dim=1)         # (batch, d+1)

    # shape (batch, d, d+1)
    H_batch = x_last.unsqueeze(2) * hz_block.unsqueeze(1)
    H_batch = H_batch.reshape(batch_size, d * (d + 1))

    return H_batch, y_last


def _sample_training_batch_factors(
    batch_size,
    d,
    ell,
    rho,
    w_task_family_train,
    device,
    dtype,
):
    """
    Generate the factorized representation of each training feature row.

    For sample n, the old feature row is
        H_n = vec(x_last[n, :, None] * hz_block[n, None, :])
    where x_last has shape (batch, d) and hz_block has shape (batch, d+1).

    Returning these factors lets us accumulate H^T H and H^T y without
    materializing H_batch of shape (batch, d * (d + 1)).
    """
    factory_kwargs = dict(device=device, dtype=dtype)

    sqrt_d = torch.sqrt(torch.tensor(float(d), **factory_kwargs))
    sqrt_rho = torch.sqrt(torch.tensor(float(rho), **factory_kwargs))

    k = w_task_family_train.shape[0]
    w_train_samples = torch.randint(0, k, (batch_size,), device=device)
    w = w_task_family_train[w_train_samples]

    s = torch.zeros(batch_size, d, **factory_kwargs)
    q = torch.zeros(batch_size, 1, **factory_kwargs)

    for _ in range(ell):
        x_t = torch.randn(batch_size, d, **factory_kwargs) / sqrt_d
        eps_t = torch.randn(batch_size, **factory_kwargs) * sqrt_rho
        y_t = torch.sum(x_t * w, dim=1) + eps_t

        s += y_t.unsqueeze(1) * x_t
        q += (y_t ** 2).unsqueeze(1)

    x_last = torch.randn(batch_size, d, **factory_kwargs) / sqrt_d
    eps_last = torch.randn(batch_size, **factory_kwargs) * sqrt_rho
    y_last = torch.sum(x_last * w, dim=1) + eps_last

    c = (float(d) / float(ell)) * s
    q = q / float(ell)
    hz_block = torch.cat([c, q], dim=1)

    return x_last, hz_block, y_last


def _accumulate_factorized_normal_equations(x_last, hz_block, y_last, solve_dtype):
    """
    Compute the chunk contribution to normal equations without H_batch.

    h_n = vec(x_n z_n^T)  has shape p = d*(d+1).
    H shape: (batch, p).

    G += H^T H   via  torch.mm(H.T, H)      — O(batch * p^2)
    b += H^T y   via  H.T @ y               — O(batch * p)

    Previously used a 4-index einsum which materialized O(batch^2 * d^2 * dz^2)
    intermediates; this version is strictly O(batch * p^2) with a single BLAS call.
    """
    x = x_last.to(dtype=solve_dtype)   # (batch, d)
    z = hz_block.to(dtype=solve_dtype)  # (batch, dz)
    y = y_last.to(dtype=solve_dtype)    # (batch,)

    batch, d = x.shape
    dz = z.shape[1]
    p = d * dz

    # H[n, :] = vec(x[n, :, None] * z[n, None, :])  — shape (batch, p)
    H = (x.unsqueeze(2) * z.unsqueeze(1)).reshape(batch, p)

    G_chunk = H.t().mm(H)          # (p, p)
    b_chunk = H.t().mv(y)          # (p,)

    return G_chunk, b_chunk

def gamma_star_torch_streaming(N, d, k, ell, rho, lam,
    strength=0.0,
    cov_type='identity',
    batch_size=128,
    device=config.DEVICE,
    sample_dtype=torch.float32,
    solve_dtype=torch.float32,
):
    """
    streaming solve for Gamma, using O(batch_size * d^2 + d^4) for the normal matrix, instead of O(N * ell * d).

    Gamma: (d, d+1) in solve_dtype
    w_task_family_train: (k, d) in sample_dtype
    w_cov
    """

    with torch.inference_mode():
        w_task_family_train, w_cov = sample_w_task_family_train(
            d=d,
            k=k,
            strength=strength,
            cov_type=cov_type,
            device=device,
            dtype=sample_dtype,
        )

        p = d * (d + 1)
        G = torch.zeros((p, p), device=device, dtype=solve_dtype)
        b = torch.zeros(p, device=device, dtype=solve_dtype)

        num_done = 0
        while num_done < N:
            cur_bs = min(batch_size, N - num_done)

            x_last, hz_block, y_last = _sample_training_batch_factors(
                batch_size=cur_bs,
                d=d,
                ell=ell,
                rho=rho,
                w_task_family_train=w_task_family_train,
                device=device,
                dtype=sample_dtype,
            )

            G_chunk, b_chunk = _accumulate_factorized_normal_equations(
                x_last=x_last,
                hz_block=hz_block,
                y_last=y_last,
                solve_dtype=solve_dtype,
            )
            G += G_chunk
            b += b_chunk

            num_done += cur_bs

            del x_last, hz_block, y_last, G_chunk, b_chunk

        reg = (N / d) * lam
        G += reg * torch.eye(p, device=device, dtype=solve_dtype)

        gamma_vec = torch.linalg.solve(G, b)
        Gamma = gamma_vec.reshape(d, d + 1)

        del G, b, gamma_vec

    return Gamma, w_task_family_train, w_cov


def _sample_test_batch_summaries(batch_size, w_task_family_train, beta, d, ell, rho,
    device=config.DEVICE,
    dtype=torch.float32,
):
    """
    Generate one test batch without materializing full x_test or y_test tensors.

    For each test context, draw a training-family index i ~ Unif([k]) and an
    independent noise direction zeta ~ N(0, Id), then set
        w_test = beta * w_i + sqrt(1 - beta^2) * zeta.

    Returns:
        H_batch: (batch_size, d, d+1)
        y_last:  (batch_size,)
    """
    factory_kwargs = dict(device=device, dtype=dtype)

    sqrt_d = torch.sqrt(torch.tensor(float(d), **factory_kwargs))
    sqrt_rho = torch.sqrt(torch.tensor(float(rho), **factory_kwargs))
    sqrt_one_minus_beta2 = torch.sqrt(
        torch.tensor(max(0.0, 1.0 - beta ** 2), **factory_kwargs)
    )

    k = w_task_family_train.shape[0]

    # Draw one training task per test context, then add fresh Gaussian noise.
    w_test_samples = torch.randint(0, k, (batch_size,), device=device)
    w_base = w_task_family_train[w_test_samples]          # (batch, d)
    zeta = torch.randn(batch_size, d, **factory_kwargs)   # fresh per context
    w_test = beta * w_base + sqrt_one_minus_beta2 * zeta  # (batch, d)

    s = torch.zeros(batch_size, d, **factory_kwargs)
    q = torch.zeros(batch_size, 1, **factory_kwargs)

    for _ in range(ell):
        x_t = torch.randn(batch_size, d, **factory_kwargs) / sqrt_d
        eps_t = torch.randn(batch_size, **factory_kwargs) * sqrt_rho
        y_t = torch.sum(x_t * w_test, dim=1) + eps_t

        s += y_t.unsqueeze(1) * x_t
        q += (y_t ** 2).unsqueeze(1)

    x_last = torch.randn(batch_size, d, **factory_kwargs) / sqrt_d
    eps_last = torch.randn(batch_size, **factory_kwargs) * sqrt_rho
    y_last = torch.sum(x_last * w_test, dim=1) + eps_last

    c = (float(d) / float(ell)) * s
    q = q / float(ell)

    hz_block = torch.cat([c, q], dim=1)          # (batch, d+1)
    H_batch = x_last.unsqueeze(2) * hz_block.unsqueeze(1)   # (batch, d, d+1)

    return H_batch, y_last


def _sample_test_batch_factors(
    batch_size,
    w_task_family_train,
    beta,
    d,
    ell,
    rho,
    device=config.DEVICE,
    dtype=torch.float32,
):
    """
    Generate factorized test features without materializing H_batch.

    Old test feature for sample n:
        H_n = x_last[n, :, None] * hz_block[n, None, :]
    Prediction:
        y_pred[n] = sum_ij Gamma[i, j] * x_last[n, i] * hz_block[n, j]
    """
    factory_kwargs = dict(device=device, dtype=dtype)

    sqrt_d = torch.sqrt(torch.tensor(float(d), **factory_kwargs))
    sqrt_rho = torch.sqrt(torch.tensor(float(rho), **factory_kwargs))
    sqrt_one_minus_beta2 = torch.sqrt(
        torch.tensor(max(0.0, 1.0 - beta ** 2), **factory_kwargs)
    )

    k = w_task_family_train.shape[0]

    w_test_samples = torch.randint(0, k, (batch_size,), device=device)
    w_base = w_task_family_train[w_test_samples]                # (batch, d)
    zeta = torch.randn(batch_size, d, **factory_kwargs)         # fresh per context
    w_test = beta * w_base + sqrt_one_minus_beta2 * zeta        # (batch, d)

    s = torch.zeros(batch_size, d, **factory_kwargs)
    q = torch.zeros(batch_size, 1, **factory_kwargs)

    for _ in range(ell):
        x_t = torch.randn(batch_size, d, **factory_kwargs) / sqrt_d
        eps_t = torch.randn(batch_size, **factory_kwargs) * sqrt_rho
        y_t = torch.sum(x_t * w_test, dim=1) + eps_t

        s += y_t.unsqueeze(1) * x_t
        q += (y_t ** 2).unsqueeze(1)

    x_last = torch.randn(batch_size, d, **factory_kwargs) / sqrt_d
    eps_last = torch.randn(batch_size, **factory_kwargs) * sqrt_rho
    y_last = torch.sum(x_last * w_test, dim=1) + eps_last

    c = (float(d) / float(ell)) * s
    q = q / float(ell)
    hz_block = torch.cat([c, q], dim=1)

    return x_last, hz_block, y_last

def test_error_torch_streaming(
    Gamma,
    N_test,
    beta,
    ell,
    rho,
    w_task_family_train,
    d,
    batch_size=256,
    device=config.DEVICE,
    sample_dtype=torch.float32,
):
    """
    Streamed test MSE.
    """
    total_se = 0.0
    total_n = 0

    with torch.inference_mode():
        Gamma_eval = Gamma.to(device=device, dtype=sample_dtype)
        w_task_family_train = w_task_family_train.to(device=device, dtype=sample_dtype)

        num_done = 0
        while num_done < N_test:
            cur_bs = min(batch_size, N_test - num_done)

            x_last, hz_block, y_last = _sample_test_batch_factors(
                batch_size=cur_bs,
                w_task_family_train=w_task_family_train,
                beta=beta,
                d=d,
                ell=ell,
                rho=rho,
                device=device,
                dtype=sample_dtype,
            )

            # (batch, d) @ (d, d+1) -> (batch, d+1); then dot with hz_block row-wise
            y_pred = (x_last @ Gamma_eval * hz_block).sum(1)
            se = torch.sum((y_pred - y_last) ** 2)

            total_se += se.item()
            total_n += cur_bs
            num_done += cur_bs

            del x_last, hz_block, y_last, y_pred, se

    return total_se / total_n


def run_single_experiment_streaming(*, alpha, beta, d, rho, kappa, tau,
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
    Wire to wire experiment
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


# === BATCHED MONTE CARLO / CLUSTER-FRIENDLY PATH ===

def sample_w_task_family_train_batched(num_runs, d, k, strength=0.0, cov_type='identity', device=config.DEVICE, dtype=torch.float32):
    """
    Sample independent train task families for multiple Monte Carlo runs at once.

    Returns:
        w_task_family_train: (num_runs, k, d)
        w_cov: covariance object/tensor from the covariance factory
    """
    if num_runs <= 0:
        raise ValueError(f"num_runs must be positive, got {num_runs}")

    families = []
    w_cov = None
    for _ in range(num_runs):
        w_task_family_train, w_cov = sample_w_task_family_train(
            d=d,
            k=k,
            strength=strength,
            cov_type=cov_type,
            device=device,
            dtype=dtype,
        )
        families.append(w_task_family_train)

    return torch.stack(families, dim=0), w_cov


def _sample_training_batch_factors_batched(
    num_runs,
    batch_size,
    d,
    ell,
    rho,
    w_task_family_train,
    device,
    dtype,
):
    """
    Factorized training features for many independent Monte Carlo runs.

    w_task_family_train: (num_runs, k, d)
    Returns:
        x_last:   (num_runs, batch_size, d)
        hz_block: (num_runs, batch_size, d + 1)
        y_last:   (num_runs, batch_size)
    """
    factory_kwargs = dict(device=device, dtype=dtype)
    sqrt_d = torch.sqrt(torch.tensor(float(d), **factory_kwargs))
    sqrt_rho = torch.sqrt(torch.tensor(float(rho), **factory_kwargs))

    k = w_task_family_train.shape[1]
    run_idx = torch.arange(num_runs, device=device).unsqueeze(1).expand(num_runs, batch_size)
    w_train_samples = torch.randint(0, k, (num_runs, batch_size), device=device)
    w = w_task_family_train[run_idx, w_train_samples]

    s = torch.zeros(num_runs, batch_size, d, **factory_kwargs)
    q = torch.zeros(num_runs, batch_size, 1, **factory_kwargs)

    for _ in range(ell):
        x_t = torch.randn(num_runs, batch_size, d, **factory_kwargs) / sqrt_d
        eps_t = torch.randn(num_runs, batch_size, **factory_kwargs) * sqrt_rho
        y_t = torch.sum(x_t * w, dim=2) + eps_t
        s += y_t.unsqueeze(2) * x_t
        q += (y_t ** 2).unsqueeze(2)

    x_last = torch.randn(num_runs, batch_size, d, **factory_kwargs) / sqrt_d
    eps_last = torch.randn(num_runs, batch_size, **factory_kwargs) * sqrt_rho
    y_last = torch.sum(x_last * w, dim=2) + eps_last

    c = (float(d) / float(ell)) * s
    q = q / float(ell)
    hz_block = torch.cat([c, q], dim=2)

    return x_last, hz_block, y_last


def _accumulate_factorized_normal_equations_batched(x_last, hz_block, y_last, solve_dtype):
    """
    Batched normal equation accumulation using bmm — no 4-index einsum.

    h_n = vec(x_n z_n^T)  has shape p = d*(d+1).
    H[r] shape: (batch, p).

    G_chunk[r] = H[r]^T H[r]   via bmm           — O(num_runs * batch * p^2)
    b_chunk[r] = H[r]^T y[r]   via batched mv     — O(num_runs * batch * p)

    The old implementation used einsum('rbi,rbj,rbk,rbl->rijkl', ...)
    which materialised an O(num_runs * batch^2 * d^2 * dz^2) intermediate —
    catastrophically expensive at large batch_size or d.
    """
    x = x_last.to(dtype=solve_dtype)   # (num_runs, batch, d)
    z = hz_block.to(dtype=solve_dtype)  # (num_runs, batch, dz)
    y = y_last.to(dtype=solve_dtype)    # (num_runs, batch)

    num_runs, batch, d = x.shape
    dz = z.shape[2]
    p = d * dz

    # Build H: (num_runs, batch, p) = vec(x[r,b,:] ⊗ z[r,b,:])
    H = (x.unsqueeze(3) * z.unsqueeze(2)).reshape(num_runs, batch, p)

    # G_chunk[r] = H[r]^T @ H[r]  — shape (num_runs, p, p)
    G_chunk = torch.bmm(H.transpose(1, 2), H)

    # b_chunk[r] = H[r]^T @ y[r]  — shape (num_runs, p)
    b_chunk = torch.bmm(H.transpose(1, 2), y.unsqueeze(2)).squeeze(2)

    return G_chunk, b_chunk


def gamma_star_torch_streaming_batched(
    num_runs,
    N,
    d,
    k,
    ell,
    rho,
    lam,
    strength=0.0,
    cov_type='identity',
    batch_size=128,
    device=config.DEVICE,
    sample_dtype=torch.float32,
    solve_dtype=torch.float32,
):
    """
    Solve num_runs independent Gamma estimates in a single batched Torch path.

    Keeps the feature matrix factorized and accumulates batched normal equations:
        G[r] = H_r.T @ H_r
        b[r] = H_r.T @ y_r
    """
    with torch.inference_mode():
        w_task_family_train, w_cov = sample_w_task_family_train_batched(
            num_runs=num_runs,
            d=d,
            k=k,
            strength=strength,
            cov_type=cov_type,
            device=device,
            dtype=sample_dtype,
        )

        p = d * (d + 1)
        G = torch.zeros((num_runs, p, p), device=device, dtype=solve_dtype)
        b = torch.zeros((num_runs, p), device=device, dtype=solve_dtype)

        num_done = 0
        while num_done < N:
            cur_bs = min(batch_size, N - num_done)

            x_last, hz_block, y_last = _sample_training_batch_factors_batched(
                num_runs=num_runs,
                batch_size=cur_bs,
                d=d,
                ell=ell,
                rho=rho,
                w_task_family_train=w_task_family_train,
                device=device,
                dtype=sample_dtype,
            )
            G_chunk, b_chunk = _accumulate_factorized_normal_equations_batched(
                x_last=x_last,
                hz_block=hz_block,
                y_last=y_last,
                solve_dtype=solve_dtype,
            )
            G += G_chunk
            b += b_chunk

            num_done += cur_bs
            del x_last, hz_block, y_last, G_chunk, b_chunk

        reg = (N / d) * lam
        eye = torch.eye(p, device=device, dtype=solve_dtype).unsqueeze(0)
        G += reg * eye

        gamma_vec = torch.linalg.solve(G, b.unsqueeze(2)).squeeze(2)
        Gamma = gamma_vec.reshape(num_runs, d, d + 1)

        del G, b, gamma_vec

    return Gamma, w_task_family_train, w_cov


def _make_batched_test_task_family(w_task_family_train, beta, d, device, dtype):
    """Create one beta-mixed test task family per Monte Carlo run.

    Args:
        w_task_family_train: Tensor with shape (num_runs, k, d).

    Returns:
        Tensor with shape (num_runs, k, d).
    """
    factory_kwargs = dict(device=device, dtype=dtype)
    sqrt_one_minus_beta2 = torch.sqrt(
        torch.tensor(max(0.0, 1.0 - beta ** 2), **factory_kwargs)
    )

    zeta = torch.randn_like(w_task_family_train, **factory_kwargs)

    return beta * w_task_family_train + sqrt_one_minus_beta2 * zeta


def _sample_test_batch_factors_batched(
    batch_size,
    w_task_family_test,
    d,
    ell,
    rho,
    device=config.DEVICE,
    dtype=torch.float32,
):
    """
    Factorized test features for many independent Monte Carlo runs.

    w_task_family_test: (num_runs, k, d)
    """
    factory_kwargs = dict(device=device, dtype=dtype)
    sqrt_d = torch.sqrt(torch.tensor(float(d), **factory_kwargs))
    sqrt_rho = torch.sqrt(torch.tensor(float(rho), **factory_kwargs))

    num_runs = w_task_family_test.shape[0]
    k = w_task_family_test.shape[1]
    run_idx = torch.arange(num_runs, device=device).unsqueeze(1).expand(num_runs, batch_size)
    w_test_samples = torch.randint(0, k, (num_runs, batch_size), device=device)
    w_test = w_task_family_test[run_idx, w_test_samples]

    s = torch.zeros(num_runs, batch_size, d, **factory_kwargs)
    q = torch.zeros(num_runs, batch_size, 1, **factory_kwargs)

    for _ in range(ell):
        x_t = torch.randn(num_runs, batch_size, d, **factory_kwargs) / sqrt_d
        eps_t = torch.randn(num_runs, batch_size, **factory_kwargs) * sqrt_rho
        y_t = torch.sum(x_t * w_test, dim=2) + eps_t
        s += y_t.unsqueeze(2) * x_t
        q += (y_t ** 2).unsqueeze(2)

    x_last = torch.randn(num_runs, batch_size, d, **factory_kwargs) / sqrt_d
    eps_last = torch.randn(num_runs, batch_size, **factory_kwargs) * sqrt_rho
    y_last = torch.sum(x_last * w_test, dim=2) + eps_last

    c = (float(d) / float(ell)) * s
    q = q / float(ell)
    hz_block = torch.cat([c, q], dim=2)

    return x_last, hz_block, y_last


def test_error_torch_streaming_batched(
    Gamma,
    N_test,
    beta,
    ell,
    rho,
    w_task_family_train,
    d,
    batch_size=256,
    device=config.DEVICE,
    sample_dtype=torch.float32,
):
    """
    Streamed test MSE for a batch of independent Monte Carlo runs.

    Gamma: (num_runs, d, d + 1)
    w_task_family_train: (num_runs, k, d)
    Returns: Python list of MSEs, length num_runs.
    """
    with torch.inference_mode():
        Gamma_eval = Gamma.to(device=device, dtype=sample_dtype)
        w_task_family_train = w_task_family_train.to(device=device, dtype=sample_dtype)
        num_runs = Gamma_eval.shape[0]

        w_task_family_test = _make_batched_test_task_family(
            w_task_family_train=w_task_family_train,
            beta=beta,
            d=d,
            device=device,
            dtype=sample_dtype,
        )

        total_se = torch.zeros(num_runs, device=device, dtype=torch.float64)
        total_n = 0
        num_done = 0
        while num_done < N_test:
            cur_bs = min(batch_size, N_test - num_done)
            x_last, hz_block, y_last = _sample_test_batch_factors_batched(
                batch_size=cur_bs,
                w_task_family_test=w_task_family_test,
                d=d,
                ell=ell,
                rho=rho,
                device=device,
                dtype=sample_dtype,
            )

            # (R, B, d) bmm (R, d, d+1) -> (R, B, d+1); dot with hz_block row-wise
            y_pred = (torch.bmm(x_last, Gamma_eval) * hz_block).sum(2)
            se = torch.sum((y_pred - y_last) ** 2, dim=1)
            total_se += se.to(dtype=torch.float64)
            total_n += cur_bs
            num_done += cur_bs

            del x_last, hz_block, y_last, y_pred, se

        mse = (total_se / total_n).detach().cpu().tolist()

    return mse


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
    Vectorized wire-to-wire experiment for multiple Monte Carlo runs.

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


# === LEGACY CODE BELOW, NOT USED IN CURRENT EXPERIMENTS, BUT KEEPING FOR REFERENCE ===

def draw_pretraining_torch(N, d, k, ell, rho, strength=0.5, cov_type='identity', device=config.DEVICE, dtype=torch.float32):
    # torch.manual_seed(seed)
    factory_kwargs = dict(device=device, dtype=dtype)

    if cov_type == 'identity':
        cov_func = corr.identity_cov
    elif cov_type == 'exp':
        cov_func = corr.exp_cov
    elif cov_type == 'powerlaw':
        cov_func = corr.powerlaw_cov

    sqrt_d = torch.sqrt(torch.tensor(float(d), **factory_kwargs))
    sqrt_rho = torch.sqrt(torch.tensor(float(rho), **factory_kwargs))

    x = torch.randn(N, ell + 1, d, **factory_kwargs) / sqrt_d

    w_cov = cov_func(d, strength, device=device, dtype=dtype)
    w_task_family_train = corr.sample_task_weights(w_cov, d, k)
    w_task_family_train = (
        w_task_family_train / (torch.linalg.norm(w_task_family_train, dim=1, keepdim=True) + 1e-12)
        * sqrt_d
    )

    w_train_samples = torch.randint(0, k, (N,), device=device)
    w = w_task_family_train[w_train_samples]  # shape (N, d)

    # y_i = x_i @ w_t + epsilon_i
    epsilon = torch.randn(N, ell + 1, **factory_kwargs) * sqrt_rho
    y = torch.einsum('nij,nj->ni', x, w) + epsilon  # shape (N, ell + 1)

    return y, x, w, w_task_family_train, w_cov, epsilon

def draw_test_torch(N_test, w_task_family_train, beta, d, ell, rho, device=config.DEVICE, dtype=torch.float32):
    # torch.manual_seed(seed)
    factory_kwargs = dict(device=device, dtype=dtype)

    sqrt_d = torch.sqrt(torch.tensor(float(d), **factory_kwargs))
    sqrt_rho = torch.sqrt(torch.tensor(float(rho), **factory_kwargs))
    sqrt_one_minus_beta2 = torch.sqrt(
        torch.tensor(1.0 - beta ** 2, **factory_kwargs)
    )

    x_test = torch.randn(N_test, ell + 1, d, **factory_kwargs) / sqrt_d
    k = w_task_family_train.shape[0]

    zeta = torch.randn(k, d, **factory_kwargs)
    proj = torch.sum(zeta * w_task_family_train, dim=1, keepdim=True) # (k, 1)
    zeta_perp = zeta - proj * w_task_family_train  # (k, d)
    zeta_perp = (
        zeta_perp / (torch.linalg.norm(zeta_perp, dim=1, keepdim=True) + 1e-12)
        * sqrt_d
    )

    w_task_family_test = beta * w_task_family_train + sqrt_one_minus_beta2 * zeta_perp
    w_test_samples = torch.randint(0, k, (N_test,), device=device)
    w_test = w_task_family_test[w_test_samples]  # shape (N_test, d)

    epsilon = torch.randn(N_test, ell + 1, **factory_kwargs) * sqrt_rho
    y_test = torch.einsum('nij,nj->ni', x_test, w_test) + epsilon

    return y_test, x_test, w_test, w_task_family_test


def H_Z_torch(y, x):
    """
    Producing H_Z. For each of N sequences, we produce d by (d + 1) feature matrix.
    """
    N, ell_plus_1, d = x.shape
    ell = ell_plus_1 - 1

    H_Z_block = torch.einsum('ni,nij->nj', y[:, :ell], x[:, :ell, :]).reshape(N, 1, d) * d / ell
    y_square_sum = torch.sum(y[:, :ell] ** 2, dim=1).reshape(N, 1, 1) / ell
    H_Z_res = torch.cat([H_Z_block, y_square_sum], dim=2)  # N by 1 by (d + 1)

    H_Z_res = H_Z_res * x[:, ell, :].reshape(N, d, 1)  # N by d by (d + 1)
    return H_Z_res


def gamma_star_torch(y, x, lam):
    """
    Solve for optimal Gamma using ridge regression logic.
    """
    N, ell_plus_1, d = x.shape
    ell = ell_plus_1 - 1

    H = H_Z_torch(y, x).reshape(N, d * (d + 1))
    y_last = y[:, ell].reshape(N)
    reg = (N / d) * lam * torch.eye(d * (d + 1), device=x.device)
    lhs = H.T @ H + reg
    rhs = H.T @ y_last

    gamma = torch.linalg.solve(lhs, rhs)
    return gamma.reshape(d, d + 1)


def test_error_torch(Gamma, N_test, beta, ell, rho, w_task_family_train, d, device=config.DEVICE):
    y_test, x_test, w_test, w_task_family_test = draw_test_torch(
        N_test, w_task_family_train, beta, d, ell, rho, device=device
    )

    H_z_test = H_Z_torch(y_test, x_test)  # N_test by d by (d + 1)
    y_pred = torch.einsum('nkl,kl->n', H_z_test, Gamma) # N_test array

    mse = torch.mean((y_pred - y_test[:, ell]) ** 2)
    return mse.item()


def run_single_experiment(*, alpha, beta, d, rho, kappa, tau,
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
    Wire-to-wire experiment using factorized HH^T streaming.

    This intentionally keeps the public nonlegacy API name but does not call
    draw_pretraining_torch(), gamma_star_torch(y, x), or test_error_torch().
    It therefore avoids materializing full training/test tensors and avoids
    materializing H_batch during the normal-equation accumulation.
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

# === LEGACY CODE BELOW, NOT USED IN CURRENT EXPERIMENTS, BUT KEEPING FOR REFERENCE ===
def Householder(beta, x):
    s = np.sign(beta[0])
    u = np.zeros_like(beta)
    u[0] = np.linalg.norm(beta) * s
    u += beta
    u /= np.linalg.norm(u)
    return -s * (x - 2 * (u.T @ x) * u)


def construct_H_NEW(beta, alpha, sigma_noise):
    d = len(beta)
    N = np.int64(alpha * d)
    theta_beta = np.linalg.norm(beta) / np.sqrt(d)
    theta_e = np.linalg.norm(np.random.randn(N)) * (sigma_noise / np.sqrt(d))
    a = np.random.randn(1)
    theta_q = np.linalg.norm(np.random.randn(N - 1)) / np.sqrt(d)
    v = np.zeros((d, 1))
    v[0] = theta_e * a / np.sqrt(d) + theta_beta * a ** 2 / d + theta_beta * theta_q ** 2
    v[1:] = np.sqrt(((theta_e + theta_beta * a / np.sqrt(d)) ** 2
                     + theta_beta ** 2 * theta_q ** 2) / d) * np.random.randn(d - 1, 1)
    av = Householder(beta, v)
    g = np.random.randn(d, 1)
    s = g[0]
    b = Householder(beta, g)
    yy = np.sqrt(1 / d) * (theta_beta ** 2 * theta_q ** 2
                           + (theta_beta * a / np.sqrt(d) + theta_e) ** 2)
    new_av = np.append(av, yy)
    y = theta_beta * s + sigma_noise * np.random.randn(1)
    return (d / N) * np.outer(b.flatten(), new_av), y


def construct_HHT_fast_NEW(beta, alpha, sigma_noise):
    n, d = beta.shape
    H = np.zeros((d * (d + 1), n))
    y_ary = np.zeros((n, 1))
    for i in range(n):
        h, y_ary[i] = construct_H_NEW(beta[i, :].reshape(d, 1), alpha, sigma_noise)
        H[:, i] = h.reshape(-1)
    return H @ H.T, H @ y_ary

def learn_Gamma_fast_NEW(beta, alpha, sigma_noise, lam, tau_max):
    n, d = beta.shape
    n_max = np.int64(tau_max * d ** 2)
    idx = np.append(np.arange(0, n, n_max), n)
    M = np.zeros((d * (d + 1), d * (d + 1)))
    v = np.zeros((d * (d + 1), 1))
    for i in range(len(idx) - 1):
        H, y = construct_HHT_fast_NEW(beta[idx[i]:idx[i + 1], :], alpha, sigma_noise)
        M += H
        v += y
    Gamma = np.linalg.solve(M + (n / d) * lam * np.eye(d * (d + 1)), v)
    return Gamma.reshape(d, d + 1)
