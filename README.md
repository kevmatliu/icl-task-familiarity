# icl-task-familiarity

Code for experiments on **task familiarity (TF)**, continuum between in-context learning and in-distribution generalization. Implements theoretical and empirical error metrics under streaming / batched Monte Carlo setups.

**Paper:** [ICL & task familiarity (PDF on Google Drive)](https://drive.google.com/file/d/1TFcAbDIrYJQ6EpYtGZYYykTR7xDSo79e/view?usp=sharing)

## Layout

| Path | Role |
|------|------|
| `main/experiments.py` | CLI / grid runs; writes JSONL under `main/results/` |
| `main/utils/` | Core numerics: sampling, streaming ridge solve for $\Gamma$, test MSE, batched MC (`run_single_experiment_batched_mc`, etc.) |
| `main/config.py` | Shared settings (e.g. default device) |
| `main/correlation.py` | Task-weight covariances and sampling for potential anisotropic task correlation matrices |