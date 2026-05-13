import torch
import correlation as corr
import config


# === LEGACY CODE BELOW, NOT USED IN CURRENT EXPERIMENTS, BUT KEEPING FOR REFERENCE ===

def draw_pretraining_torch(
    N,
    d,
    k,
    ell,
    rho,
    strength=0.5,
    cov_type='identity',
    device=config.DEVICE,
    dtype=torch.float32,
):
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
