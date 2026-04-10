import numpy as np
import torch
import correlation as corr
import config


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

    if error_type == 'IDG':
        b_train, R_train = _empirical_moments(w_task_family_train.to(device=Gamma.device, dtype=Gamma.dtype))

    ell = alpha * d
    device = Gamma.device
    dtype = Gamma.dtype

    upper_right = ((1.0 + 2.0 / ell) * (1.0 + rho) * b_train).reshape(d, 1)
    lower_left = upper_right.T
    lower_right = torch.tensor(
        [[(1.0 + 2.0 / ell) * (1.0 + rho) ** 2]], device=device, dtype=dtype
    )

    A_train = torch.cat(
        [R_train, ((1.0 + rho) * b_train).reshape(1, d)], dim=0
    )

    B_upper = torch.cat(
        [((1.0 + rho) / alpha) * torch.eye(d, device=device, dtype=dtype) + R_train, upper_right],
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
    

def draw_pretraining_torch(N, d, k, ell, rho, strength=0.5, cov_type='identity', device=config.DEVICE, dtype=torch.float64):
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

def draw_test_torch(N_test, w_task_family_train, beta, d, ell, rho, device=config.DEVICE, dtype=torch.float64):
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


if __name__ == "__main__":
    # K = 10
    # d = 5
    # B = np.random.randn(K, d)
    # rp = np.int64(np.round(0.5 * d ** 2 / K))
    # n = rp * K
    # print(f'rp: {rp}')
    # print(f'n: {n}')
    # norms = np.linalg.norm(B, axis=1)
    # B = B / norms[:, np.newaxis] * np.sqrt(d)
    # beta = np.repeat(B[np.newaxis, :, :], rp, axis=0).reshape(n, d)

    # construct_H_NEW(beta[0, :].reshape(d, 1), alpha=2, sigma_noise=1)

    # print(f'beta shape: {beta.shape}')

    # idx = np.append(np.arange(0, n, 3), n)
    # beta_chunk = beta[idx[0]:idx[1], :]
    # print(idx)
    # print(beta_chunk.shape)

    N = 1
    ell = 2
    d = 4
    k = 6
    rho = 0.5

    print('Getting training data...')
    y, x, w, w_task_family_train, epsilon = draw_pretraining_torch(N, d, k, ell, rho, cov_type='powerlaw', seed=10, device=config.DEVICE)

    print(w_task_family_train)

    print('Finished!\n')
    print('Getting H_z..')
    H_z_block = H_Z_torch(y, x)
    print(H_z_block)
    print('Finished!\n')
    Gamma_star = gamma_star_torch(y, x, lam=1e-3)
    print(Gamma_star)

    print('Getting e_icl')
    print(e_ICL_trace(Gamma_star, d, ell, rho, beta=0.5))