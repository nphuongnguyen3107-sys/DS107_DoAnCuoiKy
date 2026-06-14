# ml-pipeline/data_loading.py
"""Data loading and train/test split."""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import BINARY_COLS, CONT_COLS


def load_data(
    x_path: str = "final_rf_combined_features.csv",
    y_path: str = "y.csv",
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Load feature matrix X và label y, đồng bộ index, rồi chia train/test.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    if not os.path.exists(x_path):
        raise FileNotFoundError(f"❌ Không tìm thấy: {x_path}")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"❌ Không tìm thấy: {y_path}")

    X = pd.read_csv(x_path, index_col=0)
    print(f"📂 Features: {x_path} | Shape: {X.shape}")

    y_df = pd.read_csv(y_path, index_col=0)
    label_col = next(
        (c for c in y_df.columns if c.lower() in ["target", "label", "resistant"]),
        None,
    )
    if label_col is not None:
        y = y_df[label_col]
        print(f"✅ Nhãn cột: {label_col}")
    else:
        y = y_df.iloc[:, 0]
        print(f"⚠️ Lấy cột đầu làm nhãn: {y.name}")

    common_idx = X.index.intersection(y.index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]
    print(f"✅ Đồng bộ: X {X.shape} | y {y.shape}")

    # Xác định loại cột (chỉ trên X trước khi split)
    global BINARY_COLS, CONT_COLS
    BINARY_COLS[:] = X.columns[X.nunique() <= 2].tolist()
    CONT_COLS[:] = X.columns[X.nunique() > 2].tolist()
    print(f"   Binary cols: {len(BINARY_COLS)} | Continuous cols: {len(CONT_COLS)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    print(f"\n Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test