# ml-pipeline/config.py
"""Global configuration — all tunable hyperparameters in one place."""

import numpy as np
from sklearn.metrics import make_scorer, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold

# ── Hằng số toàn cục ────────────────────────────────────────────────────────
VAR_THRESHOLD = 0.01          # Ngưỡng loại đặc trưng phương sai thấp
TARGET_RECALL_R = 0.80        # Recall tối thiểu cho lớp Resistant
PENALTY_WEIGHT = 0.5          # Trọng số phạt khi recall < target
N_TRIALS = 50                 # Số lần thử Optuna
RANDOM_STATE = 42             # Seed tái lập

# ── Cross-validation strategy ────────────────────────────────────────────────
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

SCORING = {
    "macro_f1": make_scorer(f1_score, average="macro"),
    "recall_R": make_scorer(recall_score, pos_label=1),
}

# ── Column type lists (set after load_data) ──────────────────────────────────
BINARY_COLS: list[str] = []
CONT_COLS: list[str] = []


def soft_objective(macro_f1: float, recall_R: float,
                   target: float = TARGET_RECALL_R,
                   weight: float = PENALTY_WEIGHT) -> float:
    """Composite score: macro-F1 với penalty khi recall Resistant thấp."""
    penalty = max(0.0, target - recall_R) * weight
    return macro_f1 - penalty


def report_cv(name: str, cv_result: dict) -> float:
    """In báo cáo CV và trả về objective score."""
    mf1 = cv_result["test_macro_f1"].mean()
    rR = cv_result["test_recall_R"].mean()
    obj = soft_objective(mf1, rR)
    print(f"\n📊 {name}:")
    print(f"  Macro F1 : {mf1*100:.2f}%")
    print(f"  Recall(R) : {rR*100:.2f}%")
    print(f"  Objective : {obj:.4f}")
    return obj