import torch
import config


def make_psd(W):
    # ensuring matrix is positive semi-definite for valid covariance matrix
    e_val, e_vec = torch.linalg.eigh(W)
    
    return e_vec @ torch.diag(torch.clamp(e_val, min=1e-10)) @ e_vec.T


def identity_cov(d, strength, device=config.DEVICE, dtype=torch.float64):
    return torch.eye(d, device=device, dtype=dtype)


def exp_cov(d, strength, device=config.DEVICE, dtype=torch.float64):
    # task diversity K
    idx = torch.arange(d, device=device, dtype=dtype)
    W = strength ** torch.abs(idx[:, None] - idx[None, :])
    return make_psd(W)


def powerlaw_cov(d, strength, device=config.DEVICE, dtype=torch.float64):
    idx = torch.arange(d, device=device, dtype=dtype)

    W = 1.0 / (1.0 + torch.abs(idx[:, None] - idx[None, :])) ** strength
    return make_psd(W)


def sample_task_weights(W, d, k, dtype=torch.float64):
    e_val, e_vec = torch.linalg.eigh(W)
    W_sqrt = e_vec @ torch.diag(torch.sqrt(torch.clamp(e_val, min=0))) @ e_vec.T
    Z = torch.randn(k, d, device=W.device, dtype=dtype)

    W_new = Z @ W_sqrt.T    # k x d
    # W_new = (d ** 0.5) * W_new / (torch.linalg.norm(W_new, dim=1, keepdim=True) + 1e-12)

    return W_new


if __name__ == '__main__':
    d = 10
    strength = 0.5
    W = exp_cov(d, strength)
    print(W)