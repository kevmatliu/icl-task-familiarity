import torch


def make_psd(W):
    # ensuring matrix is positive semi-definite for valid covariance matrix
    e_val, e_vec = torch.linalg.eigh(W)
    
    return e_vec @ torch.diag(torch.clamp(e_val, min=1e-10)) @ e_vec.T


def identity_cov(K, strength, device='cuda', dtype=torch.float64):
    return torch.eye(K, device=device, dtype=dtype)


def exp_cov(K, strength, device='cuda', dtype=torch.float64):
    # task diversity K
    idx = torch.arange(K, device=device, dtype=dtype)
    W = strength ** torch.abs(idx[:, None] - idx[None, :])
    return make_psd(W)


def powerlaw_cov(K, strength, device='cuda', dtype=torch.float64):
    idx = torch.arange(K, device=device, dtype=dtype)

    W = 1.0 / (1.0 + torch.abs(idx[:, None] - idx[None, :])) ** strength
    return make_psd(W)


def sample_task_weights(W, d, k, dtype=torch.float64):
    if k is None:
        k = d
        
    e_val, e_vec = torch.linalg.eigh(W)
    W_sqrt = e_vec @ torch.diag(torch.sqrt(torch.clamp(e_val, min=0))) @ e_vec.T
    Z = torch.randn(k, d, device=W.device, dtype=dtype)

    W_new = W_sqrt @ Z
    W_new = (d ** 0.5) * W_new / (torch.linalg.norm(W_new, dim=1, keepdim=True) + 1e-12)

    return W_new


if __name__ == '__main__':
    K = 10
    strength = 0.5
    W = exp_cov(K, strength)
    print(W)