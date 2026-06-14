# ml-pipeline/training.py
"""Model training — Optuna tuning + stacking ensemble."""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.base import clone
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, RFE
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

from .config import (
    VAR_THRESHOLD, RANDOM_STATE, N_TRIALS, TARGET_RECALL_R,
    cv_strategy, SCORING, soft_objective, report_cv,
)
from .inference import find_best_threshold_inner


# ── Pipeline builders ────────────────────────────────────────────────────────

def build_xgb_pipeline(params: dict, k_features: int, smote_strategy: float) -> ImbPipeline:
    base = XGBClassifier(**params, random_state=RANDOM_STATE, eval_metric="logloss")
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("var_thresh", VarianceThreshold(threshold=VAR_THRESHOLD)),
        ("rfe", RFE(estimator=base, n_features_to_select=min(k_features, 310), step=20)),
        ("smote", SMOTE(sampling_strategy=smote_strategy, random_state=RANDOM_STATE)),
    ]
    clf_params = {**params, "random_state": RANDOM_STATE, "n_jobs": 1}
    steps.append(("clf", XGBClassifier(**clf_params, eval_metric="logloss")))
    return ImbPipeline(steps)


def build_rf_pipeline(params: dict, k_features: int, smote_strategy: float) -> ImbPipeline:
    base = RandomForestClassifier(**params, random_state=RANDOM_STATE)
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("var_thresh", VarianceThreshold(threshold=VAR_THRESHOLD)),
        ("rfe", RFE(estimator=base, n_features_to_select=min(k_features, 310), step=20)),
        ("smote", SMOTE(sampling_strategy=smote_strategy, random_state=RANDOM_STATE)),
    ]
    clf_params = {**params, "random_state": RANDOM_STATE, "n_jobs": 1}
    steps.append(("clf", RandomForestClassifier(**clf_params)))
    return ImbPipeline(steps)


def build_lgbm_pipeline(params: dict, k_features: int, smote_strategy: float) -> ImbPipeline:
    base = LGBMClassifier(**params, random_state=RANDOM_STATE, verbose=-1)
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("var_thresh", VarianceThreshold(threshold=VAR_THRESHOLD)),
        ("rfe", RFE(estimator=base, n_features_to_select=min(k_features, 310), step=20)),
        ("smote", SMOTE(sampling_strategy=smote_strategy, random_state=RANDOM_STATE)),
    ]
    clf_params = {**params, "random_state": RANDOM_STATE, "n_jobs": 1}
    steps.append(("clf", LGBMClassifier(**clf_params)))
    return ImbPipeline(steps)


# ── Optuna objectives ────────────────────────────────────────────────────────
# FIX LỖI 3: nhận X_train, y_train làm parameter (closure), không dùng global

def objective_xgb(trial, X_train, y_train):
    np.random.seed(RANDOM_STATE)
    smote_strategy = trial.suggest_float("smote_strategy", 0.58, 1.0)
    k_features = trial.suggest_int("k_features", 10, min(100, 310))
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 2, 5),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 3.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }
    try:
        pipe = build_xgb_pipeline(params, k_features, smote_strategy)
        scores = cross_validate(pipe, X_train, y_train, cv=cv_strategy, scoring=SCORING, n_jobs=-1)
        return soft_objective(scores["test_macro_f1"].mean(), scores["test_recall_R"].mean())
    except Exception:
        return 0.0


def objective_rf(trial, X_train, y_train):
    np.random.seed(RANDOM_STATE)
    smote_strategy = trial.suggest_float("smote_strategy", 0.58, 1.0)
    k_features = trial.suggest_int("k_features", 10, min(100, 310))
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced_subsample", None]),
    }
    try:
        pipe = build_rf_pipeline(params, k_features, smote_strategy)
        scores = cross_validate(pipe, X_train, y_train, cv=cv_strategy, scoring=SCORING, n_jobs=-1)
        return soft_objective(scores["test_macro_f1"].mean(), scores["test_recall_R"].mean())
    except Exception:
        return 0.0


def objective_lgbm(trial, X_train, y_train):
    np.random.seed(RANDOM_STATE)
    smote_strategy = trial.suggest_float("smote_strategy", 0.58, 1.0)
    k_features = trial.suggest_int("k_features", 10, min(100, 310))
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 3.0),
    }
    try:
        pipe = build_lgbm_pipeline(params, k_features, smote_strategy)
        scores = cross_validate(pipe, X_train, y_train, cv=cv_strategy, scoring=SCORING, n_jobs=-1)
        return soft_objective(scores["test_macro_f1"].mean(), scores["test_recall_R"].mean())
    except Exception:
        return 0.0


def build_stacking_ensemble(pipe_xgb, pipe_rf, pipe_lgbm) -> StackingClassifier:
    """Xây dựng Stacking Classifier từ các model cơ bản."""
    # Stacking dùng cv=3 để tăng tốc
    stacking_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    return StackingClassifier(
        estimators=[
            ('xgb', pipe_xgb),
            ('rf', pipe_rf),
            ('lgbm', pipe_lgbm),
        ],
        final_estimator=LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            solver='lbfgs',
        ),
        cv=stacking_cv,
        n_jobs=1,
        passthrough=False,
    )


def find_optimal_threshold(clf, X_train, y_train) -> float:
    """Tìm threshold tối ưu bằng out-of-fold predictions sử dụng cross_val_predict."""
    from sklearn.model_selection import cross_val_predict
    from .inference import find_best_threshold_inner
    
    # Lấy xác suất dự đoán out-of-fold cho toàn bộ X_train song song
    y_proba_oof = cross_val_predict(
        clf, X_train, y_train, 
        cv=cv_strategy, 
        method="predict_proba", 
        n_jobs=-1
    )[:, 1]
    
    threshold, _ = find_best_threshold_inner(y_train, y_proba_oof, target_recall=TARGET_RECALL_R)
    return threshold


# ── High-level training functions ───────────────────────────────────────────

def train_all_models(X_train, y_train, n_trials: int = N_TRIALS):
    """
    Chạy Optuna cho cả 3 model, fit trên toàn bộ X_train, trả về fitted pipelines.

    Returns
    -------
    dict với keys: xgb, rf, lgbm, stacking
    Mỗi item là (pipeline_fitted, best_params_dict, threshold)
    """
    import optuna

    print("=" * 60)
    print("🔍 OPTUNA TUNING — 3 MODELS")
    print("=" * 60)

    best_configs = {}

    for name, objective in [
        ("XGBoost", objective_xgb),
        ("RandomForest", objective_rf),
        ("LightGBM", objective_lgbm),
    ]:
        print(f"\n── {name} ──")
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        )
        actual_trials = n_trials if name == "XGBoost" else 5
        study.optimize(
            lambda t: objective(t, X_train, y_train),  # FIX LỖI 3: truyền data vào
            n_trials=actual_trials,
            show_progress_bar=True,
        )
        best_params = {
            k: v for k, v in study.best_params.items()
            if k not in ("smote_strategy", "k_features")
        }
        smote = study.best_params["smote_strategy"]
        k_feats = study.best_params["k_features"]
        print(f" Best trial: {study.best_trial.number} | value: {study.best_value:.4f}")
        best_configs[name] = (best_params, k_feats, smote)  # (params, k_feats, smote)

    # FIX LỖI 1: unpack đúng thứ tự — (params, k_features, smote_strategy)
    pipe_xgb = build_xgb_pipeline(*best_configs["XGBoost"])
    pipe_rf  = build_rf_pipeline(*best_configs["RandomForest"])
    pipe_lgbm = build_lgbm_pipeline(*best_configs["LightGBM"])

    print("\n📊 Cross-Validation...")
    for name, pipe in [("XGBoost", pipe_xgb), ("RF", pipe_rf), ("LGBM", pipe_lgbm)]:
        cv_res = cross_validate(pipe, X_train, y_train, cv=cv_strategy, scoring=SCORING, n_jobs=-1)
        report_cv(name, cv_res)

    # FIX LỖI 2: fit trên toàn bộ data trước khi trả về
    print("\n Fitting trên toàn bộ X_train...")
    pipe_xgb.fit(X_train, y_train)
    pipe_rf.fit(X_train, y_train)
    pipe_lgbm.fit(X_train, y_train)

    # Stacking
    print("\n📦 Building stacking ensemble...")
    stacking = build_stacking_ensemble(pipe_xgb, pipe_rf, pipe_lgbm)
    stacking.fit(X_train, y_train)

    # Tìm threshold cho từng model
    th_xgb  = find_optimal_threshold(pipe_xgb, X_train, y_train)
    th_rf   = find_optimal_threshold(pipe_rf, X_train, y_train)
    th_lgbm = find_optimal_threshold(pipe_lgbm, X_train, y_train)
    th_ens  = find_optimal_threshold(stacking, X_train, y_train)

    return {
        "xgb":     (pipe_xgb,   best_configs["XGBoost"],    th_xgb),
        "rf":      (pipe_rf,    best_configs["RandomForest"], th_rf),
        "lgbm":    (pipe_lgbm,  best_configs["LightGBM"],    th_lgbm),
        "stacking": (stacking,   {},                          th_ens),
    }