import os
import sys

# Chuyển thư mục làm việc về thư mục chứa script để tránh lỗi đường dẫn tương đối
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Đảm bảo stdout/stderr sử dụng UTF-8 để tránh UnicodeEncodeError trên Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import warnings
warnings.filterwarnings('ignore')

import ml_pipeline

def main():
    # 1. Tải và phân chia dữ liệu
    print("1. Đang tải dữ liệu...")
    X_train, X_test, y_train, y_test = ml_pipeline.load_data(
        x_path='X.csv', 
        y_path='y.csv'
    )

    # 2. Huấn luyện và tối ưu hóa hyperparameter bằng Optuna (50 trials)
    # Lưu ý: Quá trình này có thể mất vài phút vì chạy tìm kiếm trên 3 thuật toán
    print("\n2. Bắt đầu huấn luyện và tối ưu hóa các mô hình (XGBoost, RF, LightGBM, Stacking)...")
    results = ml_pipeline.train_all_models(X_train, y_train, n_trials=50)

    # 3. Lấy mô hình XGBoost Pipeline và threshold tối ưu của nó
    xgb_pipeline, _, threshold = results["xgb"]
    features = X_train.columns.tolist()

    # 4. Lưu mô hình xuống file .joblib
    print("\n3. Đang lưu mô hình tối ưu...")
    model_path = ml_pipeline.save_model(
        model=xgb_pipeline,
        threshold=threshold,
        features=features,
        model_name="amr_classifier"
    )

    print(f"\n✅ Hoàn thành huấn luyện! File mô hình được lưu tại: {model_path}")
    print("Bạn có thể dùng file .joblib này để tích hợp vào Web App.")

    # 5. Đánh giá mô hình trên tập Test (Unseen Data) để kiểm tra độ chính xác
    print("\n=======================================================")
    print(" EVALUATION ON TEST SET (UNSEEN DATA)")
    print("=======================================================")
    from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

    y_proba_test = xgb_pipeline.predict_proba(X_test)[:, 1]
    y_pred_test = (y_proba_test >= threshold).astype(int)

    print(f"Test set size: {len(X_test)}")
    print(f"Threshold used: {threshold:.3f}\n")
    print(classification_report(y_test, y_pred_test, target_names=["Susceptible", "Resistant"]))

    roc_auc = roc_auc_score(y_test, y_proba_test)
    ap_score = average_precision_score(y_test, y_proba_test)
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC (Average Precision): {ap_score:.4f}")
    print("=======================================================")

if __name__ == "__main__":
    main()
