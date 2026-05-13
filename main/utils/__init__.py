"""
Experiment utilities split into focused submodules.

Import from the package root for the same API as the former ``utils.py``:

    from utils import run_single_experiment_batched_mc
"""

from .tensor import convert_tensor_scalar
from .error_metrics import e_, diff_matrix, e_TF
from .sampling import sample_w_task_family_train, sample_w_task_family_train_batched
from .batches_train import (
    gamma_star_torch_streaming,
    gamma_star_torch_streaming_batched,
    _sample_training_batch_summaries,
    _sample_training_batch_factors,
    _accumulate_factorized_normal_equations,
    _sample_training_batch_factors_batched,
    _accumulate_factorized_normal_equations_batched,
)
from .batches_test import (
    test_error_torch_streaming,
    test_error_torch_streaming_batched,
    _sample_test_batch_summaries,
    _sample_test_batch_factors,
    _sample_test_batch_factors_batched,
    _make_batched_test_task_family,
)
from .experiment_runs import (
    run_single_experiment_streaming,
    run_single_experiment_batched_mc,
    run_single_experiment,
)
from .legacy_torch import (
    draw_pretraining_torch,
    draw_test_torch,
    H_Z_torch,
    gamma_star_torch,
    test_error_torch,
)
from .legacy_numpy import (
    Householder,
    construct_H_NEW,
    construct_HHT_fast_NEW,
    learn_Gamma_fast_NEW,
)

__all__ = [
    'convert_tensor_scalar',
    'e_',
    'diff_matrix',
    'e_TF',
    'sample_w_task_family_train',
    'sample_w_task_family_train_batched',
    'gamma_star_torch_streaming',
    'gamma_star_torch_streaming_batched',
    'test_error_torch_streaming',
    'test_error_torch_streaming_batched',
    'run_single_experiment_streaming',
    'run_single_experiment_batched_mc',
    'run_single_experiment',
    'draw_pretraining_torch',
    'draw_test_torch',
    'H_Z_torch',
    'gamma_star_torch',
    'test_error_torch',
    'Householder',
    'construct_H_NEW',
    'construct_HHT_fast_NEW',
    'learn_Gamma_fast_NEW',
    '_sample_training_batch_summaries',
    '_sample_training_batch_factors',
    '_accumulate_factorized_normal_equations',
    '_sample_training_batch_factors_batched',
    '_accumulate_factorized_normal_equations_batched',
    '_sample_test_batch_summaries',
    '_sample_test_batch_factors',
    '_sample_test_batch_factors_batched',
    '_make_batched_test_task_family',
]
