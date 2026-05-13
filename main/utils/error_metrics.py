import torch


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
