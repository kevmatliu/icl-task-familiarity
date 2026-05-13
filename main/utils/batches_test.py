import torch
import config


def _sample_test_batch_summaries(
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
