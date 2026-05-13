import torch
import config

from .sampling import sample_w_task_family_train, sample_w_task_family_train_batched


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


def gamma_star_torch_streaming(
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
