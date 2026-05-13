import numpy as np


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
