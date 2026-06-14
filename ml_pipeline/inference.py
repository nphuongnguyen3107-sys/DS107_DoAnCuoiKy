# ml-pipeline/inference.py
"""Model persistence and single-patient inference."""

import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import f1_score, recall_score

from .config import TARGET_RECALL_R


def save_model(
    model,
    threshold: float,
    features: list[str],
    model_name: str = "amr_classifier",
) -> str:
    """Lưu model + threshold + feature list vào file joblib."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"{model_name}_{timestamp}.joblib"
    joblib.dump(
        {
            "model": model,
            "threshold": threshold,
            "features": features,
            "created_at": timestamp,
        },
        model_path,
    )
    print(f"💾 Saved: {model_path}")
    return model_path


def load_model(model_path: str) -> tuple:
    """Load model từ file joblib. Returns (model, threshold, features)."""
    data = joblib.load(model_path)
    return data["model"], data["threshold"], data["features"]


def find_best_threshold_inner(
    y_true: pd.Series,
    y_proba: np.ndarray,
    target_recall: float = TARGET_RECALL_R,
) -> tuple[float, float]:
    """
    Tìm threshold tốt nhất: ưu tiên recall Resistant >= target,
    rồi chọn threshold có macro-F1 cao nhất trong số đó.
    """
    thresholds = np.arange(0.30, 0.71, 0.001)
    rows = []
    for th in thresholds:
        y_p = (y_proba >= th).astype(int)
        rows.append({
            "threshold": th,
            "macro_f1": f1_score(y_true, y_p, average="macro"),
            "recall_R": recall_score(y_true, y_p, pos_label=1),
        })
    df = pd.DataFrame(rows)
    candidates = df[df["recall_R"] >= target_recall]
    if len(candidates) > 0:
        best = candidates.sort_values("macro_f1", ascending=False).iloc[0]
    else:
        best = df.sort_values("macro_f1", ascending=False).iloc[0]
    return float(best["threshold"]), float(best["macro_f1"])


def predict_one_patient(
    feature_vector: pd.Series,
    model,
    threshold: float,
    expected_features: list[str],
) -> dict:
    """
    Dự đoán cho 1 bệnh nhân mới.

    Parameters
    ----------
    feature_vector : Series với đúng 310 feature columns (theo thứ tự khi train)
    model : trained pipeline (ImbPipeline)
    threshold : ngưỡng quyết định
    expected_features : list tên cột khi train (để đảm bảo thứ tự)

    Returns
    -------
    dict với prediction, confidence, prob_resistant
    """
    # Đảm bảo đúng thứ tự cột như khi train
    X = feature_vector.reindex(expected_features).values.reshape(1, -1)
    proba = model.predict_proba(X)[0, 1]
    prediction = "Resistant" if proba >= threshold else "Susceptible"
    return {
        "prediction": prediction,
        "prob_resistant": round(float(proba), 4),
        "confidence": round(float(max(proba, 1 - proba)), 4),
        "threshold_used": threshold,
    }