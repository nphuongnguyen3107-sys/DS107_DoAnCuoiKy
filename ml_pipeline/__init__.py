# ml-pipeline/__init__.py
"""AMR Classification Pipeline — reusable ML package."""

from .config import (
    VAR_THRESHOLD, TARGET_RECALL_R, PENALTY_WEIGHT,
    N_TRIALS, RANDOM_STATE, BINARY_COLS, CONT_COLS,
    cv_strategy, SCORING, soft_objective, report_cv,
)
from .data_loading import load_data
from .inference import (
    save_model, load_model, predict_one_patient,
    find_best_threshold_inner,
)
from .training import (
    build_xgb_pipeline, build_rf_pipeline, build_lgbm_pipeline,
    train_all_models, build_stacking_ensemble, find_optimal_threshold,
)
from .explain import build_shap_explainer, explain_prediction

__all__ = [
    "VAR_THRESHOLD", "TARGET_RECALL_R", "PENALTY_WEIGHT",
    "N_TRIALS", "RANDOM_STATE", "BINARY_COLS", "CONT_COLS",
    "cv_strategy", "SCORING",
    "load_data",
    "save_model", "load_model", "predict_one_patient",
    "find_best_threshold_inner", "soft_objective", "report_cv",
    "build_xgb_pipeline", "build_rf_pipeline", "build_lgbm_pipeline",
    "train_all_models", "build_stacking_ensemble", "find_optimal_threshold",
    "build_shap_explainer", "explain_prediction",
]