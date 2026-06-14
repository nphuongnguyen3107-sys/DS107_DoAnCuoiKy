# ml-pipeline/explain.py
"""SHAP explanations — interpretable predictions for clinicians."""

import numpy as np
import pandas as pd
import shap


def build_shap_explainer(model, X_background: pd.DataFrame):
    """
    Tạo SHAP explainer cho model.

    Parameters
    ----------
    model : trained ImbPipeline
    X_background : ~100 mẫu để làm baseline (shap cần background distribution)

    Returns
    -------
    shap.explainers.Permutation
    """
    # Sử dụng Permutation explainer trực tiếp để đảm bảo cấu hình max_evals được áp dụng chính xác
    num_features = X_background.shape[1]
    max_evals_val = max(500, 2 * num_features + 500)
    explainer = shap.explainers.Permutation(model.predict_proba, X_background, max_evals=max_evals_val)
    return explainer


def explain_prediction(
    explainer,
    feature_vector: pd.Series,
    expected_features: list[str],
    top_k: int = 10,
) -> dict:
    """
    Giải thích 1 dự đoán — trả về top features đẩy prediction lên Resistant.

    Returns
    -------
    dict với keys: top_features (list[dict]), base_value, prediction_value
    """
    import math
    X = feature_vector.reindex(expected_features).values.reshape(1, -1)
    # Tăng max_evals để tránh lỗi với bộ giải thích Permutation của SHAP khi số lượng đặc trưng lớn
    shap_values = explainer(X, max_evals=2 * len(expected_features) + 500)
    # shap_values.values shape: (1, 2, n_features) → lấy class 1 (Resistant)
    vals = shap_values.values[0, :, 1]
    cols = np.array(expected_features)

    # Top features đẩy prediction lên (SHAP value dương = tăng risk)
    top_idx = np.argsort(vals)[::-1][:top_k]
    top_features = []
    for i in top_idx:
        feat_val = X[0, i]
        # Nếu là NaN (giá trị trống), đổi thành None để khi chuyển thành JSON sẽ là null, tránh lỗi SyntaxError ở Frontend
        if pd.isna(feat_val) or (isinstance(feat_val, float) and math.isnan(feat_val)):
            feat_val_json = None
        else:
            feat_val_json = round(float(feat_val), 4)

        top_features.append({
            "feature": str(cols[i]),
            "shap_value": round(float(vals[i]), 4),
            "feature_value": feat_val_json
        })

    base_val = float(shap_values.base_values[0, 1])
    pred_val = float(vals.sum() + shap_values.base_values[0, 1])

    return {
        "top_features": top_features,
        "base_value": round(base_val, 4) if not math.isnan(base_val) else None,
        "prediction_value": round(pred_val, 4) if not math.isnan(pred_val) else None,
    }