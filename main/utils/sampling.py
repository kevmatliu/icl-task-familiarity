import torch
import correlation as corr
import config


def sample_w_task_family_train(
    d,
    k,
    strength=0.0,
    cov_type='identity',
    device=config.DEVICE,
    dtype=torch.float32,
):
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


def sample_w_task_family_train_batched(
    num_runs,
    d,
    k,
    strength=0.0,
    cov_type='identity',
    device=config.DEVICE,
    dtype=torch.float32,
):
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
