# Experiments subpackage for PA-SSL diagnostic scripts

from .runner import (
    add_electrode_noise,
    add_motion_artifact,
    robustness_experiment,
    ood_detection_experiment,
    run_ablation_suite,
    cross_dataset_transfer_matrix,
    per_augmentation_ablation
)

# Phase 1: Multi-task clinical evaluation
from .multi_task_evaluation import (
    AgeRegressionProbe,
    SexClassificationProbe,
    ECGAgeGapAnalyzer,
    EmbeddingRetrievalEvaluator,
    run_multi_task_evaluation,
)

# Phase 2: Scaling laws (import lazily to avoid heavy deps at import time)
def run_scaling_experiment(*args, **kwargs):
    from .scaling_laws import run_scaling_experiment as _run
    return _run(*args, **kwargs)
