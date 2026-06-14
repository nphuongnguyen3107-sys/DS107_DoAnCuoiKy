# 🔬 E. coli Ciprofloxacin AMR Analyzer & Explainer
## Hệ Thống Dự Đoán & Giải Thích Kháng Thuốc Ciprofloxacin ở Vi Khuẩn *Escherichia coli*

Dự án đồ án môn DS107 (Tư duy Khoa học dữ liệu) xây dựng hệ thống học máy khép kín dự đoán kiểu hình kháng thuốc kháng sinh **Ciprofloxacin** (nhóm Fluoroquinolone) của vi khuẩn *E. coli* dựa trên dữ liệu hệ gen (gen kháng thuốc và mật độ k-mer) kết hợp các kỹ thuật giải thích mô hình (SHAP) phục vụ nghiên cứu sinh tin học và vi sinh lâm sàng.

---

## 📂 Cấu Trúc Dự Án

Dự án được sắp xếp và tổ chức lại một cách khoa học theo cấu trúc chuẩn hóa dưới đây:

```text
Đồ Án DS107 (Tư duy Khoa học dữ liệu)/ 
├── data/                            # Thư mục chứa dữ liệu đầu vào học máy
│   ├── X.csv                        # Ma trận đặc trưng (2404 mẫu × 310 đặc trưng)
│   └── y.csv                        # Nhãn kiểu hình kháng thuốc (Resistant/Susceptible)
├── models/                          # Thư mục lưu trữ mô hình và cơ sở dữ liệu lịch sử
│   ├── amr_classifier_*.joblib      # Các file mô hình đã huấn luyện (đã tối ưu bằng Optuna)
│   └── amr_history.db               # Cơ sở dữ liệu SQLite lưu lịch sử chẩn đoán
├── ml_pipeline/                     # Các mô-đun Python cốt lõi của pipeline ML
│   ├── config.py                    # Cấu hình siêu tham số, chiến lược CV và random seed
│   ├── data_loading.py              # Đọc dữ liệu đầu vào, phân tách Train/Test
│   ├── training.py                  # Logic tối ưu hóa bằng Optuna và xây dựng Stacking/XGBoost
│   ├── inference.py                 # Hàm dự đoán đơn mẫu, xuất kết quả kiểu hình
│   └── explain.py                   # Tích hợp SHAP sinh giá trị đóng góp đặc trưng cục bộ
├── preprocessing/                   # Thư mục tiền xử lý gen và k-mer (đã phẳng hóa)
│   ├── data/                        # Dữ liệu raw và processed trung gian của tiền xử lý
│   │   ├── raw/BVBRC_genome_amr.csv # Metadata kiểu hình từ BV-BRC
│   │   └── processed/               # Các ma trận đặc trưng trung gian thu được
│   ├── scripts/                     # Các script tiền xử lý độc lập (AMRFinder, K-mer, RFE)
│   │   ├── run_amr.py               # Chạy AMRFinderPlus quét gen kháng từ FASTA thô
│   │   ├── merge_dataset.py         # Khớp mã mẫu gen với nhãn kiểu hình lâm sàng
│   │   ├── K-mer.py                 # Trích xuất k-mer amino acid từ protein sequence
│   │   └── select_top100_aa_kmers.py# Chọn lọc k-mer bằng Random Forest Importance
│   └── PREPROCESSING.md             # Báo cáo kỹ thuật chi tiết về quy trình tiền xử lý
├── static/                          # Các tài nguyên tĩnh giao diện (CSS, JS, Fonts)
│   ├── css/style.css                # CSS giao diện Glassmorphism hiện đại
│   └── js/app.js                    # Logic Frontend xử lý AJAX, vẽ biểu đồ Chart.js
├── templates/                       # Thư mục chứa giao diện HTML
│   └── index.html                   # Giao diện Dashboard chính (HTML5 + Tailwind-free)
├── uploads/                         # Thư mục tạm nhận file CSV upload hàng loạt
├── run_web_app.py                           # Điểm chạy Web App Flask chính
├── run_training.py                  # Script huấn luyện và tối ưu mô hình tổng thể
├── run_evaluation.py                # Script chạy 4 phân tích đánh giá sâu mô hình
├── .env                             # Cấu hình biến môi trường và API Keys (Gemini)
├── .env.example                     # Mẫu cấu hình biến môi trường
├── .gitignore                       # Cấu hình bỏ qua các file nặng/nhạy cảm trong Git
└── requirements.txt                 # Danh sách các thư viện Python cần thiết
```

---

## 🛠️ Công Nghệ Sử Dụng

* **Ngôn ngữ chính**: Python (v3.10+)
* **Học máy & Tối ưu hóa**: scikit-learn, XGBoost, LightGBM, Imbalanced-learn (SMOTE), Optuna
* **Giải thích mô hình (Explainable AI)**: SHAP (Shapley Additive exPlanations)
* **Back-end Web**: Flask (Python)
* **CSDL & Lưu trữ**: SQLite, Joblib
* **Giao diện người dùng (Frontend)**: Vanilla HTML5 & CSS3, JavaScript (ES6+), Chart.js (vẽ biểu đồ động), FontAwesome (icons)

---

## 📈 Quy Trình Pipeline Học Máy Đề Xuất (XGBoost Pipeline)

Hệ thống được thiết kế khép kín nhằm chống rò rỉ dữ liệu (data leakage) trong quá trình đánh giá:

1. **Median Imputer**: Xử lý dữ liệu k-mer khuyết bằng phương pháp điền trung vị.
2. **Variance Threshold**: Loại bỏ các đặc trưng tĩnh hoặc biến động cực thấp ($< 0.01$).
3. **RFE (Recursive Feature Elimination)**: Lọc giảm chiều đặc trưng, chỉ giữ lại **93 đặc trưng gen/k-mer quan trọng nhất** có khả năng phân biệt cao nhằm tối ưu chi phí giải trình tự gen.
4. **SMOTE (Synthetic Minority Over-sampling Technique)**: Cân bằng tỉ lệ mẫu kháng thuốc (thiểu số), giúp tăng mạnh Recall lớp kháng từ **55.63% lên 80.82%**.
5. **XGBoost Classifier**: Mô hình phân loại tối ưu sau khi chạy tìm kiếm siêu tham số bằng Optuna qua 50 thử nghiệm.
6. **Decision Threshold (0.521)**: Ngưỡng ra quyết định được tinh chỉnh out-of-fold nhằm đảm bảo độ nhạy lâm sàng tối thiểu (Recall lớp kháng $\ge 80\%$).

### Chỉ số hiệu năng trên tập kiểm thử (Test Set)
* **Độ chính xác (Accuracy)**: **83.00%**
* **ROC-AUC**: **90.29%**
* **PR-AUC**: **89.05%**

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy Hệ Thống

### 1. Chuẩn bị môi trường Python
Yêu cầu Python từ 3.10 trở lên. Cài đặt môi trường ảo và cài thư viện:

```bash
# Tạo môi trường ảo (Virtual Environment)
python -m venv venv

# Kích hoạt môi trường ảo
# Trên Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Trên Linux/macOS:
source venv/bin/activate

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

### 2. Thiết lập cấu hình biến môi trường
Tạo file `.env` từ file mẫu `.env.example`:
```bash
copy .env.example .env
```
Nếu bạn muốn sử dụng tính năng **Báo cáo AI & Trợ lý tư vấn AMR thông minh tự động**, hãy điền `GEMINI_API_KEY` của bạn vào file `.env`. Nếu không có key, hệ thống sẽ tự động hoạt động ở chế độ ngoại tuyến (Offline Local Expert Mode) dựa trên bộ luật cơ sở dữ liệu vi sinh có sẵn.

### 3. Huấn luyện mô hình (Tùy chọn)
Nếu bạn muốn huấn luyện lại mô hình với dữ liệu trong thư mục `data/`:
```bash
python run_training.py
```
*Script này sẽ chạy tìm kiếm siêu tham số Optuna cho cả 3 mô hình XGBoost, Random Forest, LightGBM và tự động lưu mô hình XGBoost tối ưu nhất cùng bộ tham số chuẩn vào thư mục `models/`*.

### 4. Đánh giá chuyên sâu mô hình
Chạy đánh giá sâu mô hình phục vụ báo cáo đồ án (Baseline, Top 20 đặc trưng quan trọng, Ablation study, Ma trận nhầm lẫn y tế):
```bash
python run_evaluation.py
```

### 5. Chạy giao diện Web App
Khởi chạy ứng dụng Dashboard trực quan:
```bash
python run_web_app.py
```
Truy cập ứng dụng tại địa chỉ: `http://127.0.0.1:5000` trên trình duyệt web của bạn.

---

## 🖥️ Các Tính Năng Chính Của Web App

1. **Dashboard Tổng Quan**: Hiển thị các chỉ số chất lượng mô hình (ROC-AUC, PR-AUC, Accuracy) cùng sơ đồ quy trình ML Pipeline và bảng so sánh các phương pháp (Baseline vs. Stacking vs. XGBoost).
2. **Chẩn Đoán Đơn Chủng (Single Diagnostic)**: 
   * Hỗ trợ tải mẫu có sẵn hoặc nhập dữ liệu JSON đặc trưng gen của chủng vi khuẩn cần chẩn đoán.
   * Hiển thị kết luận nhãn (Kháng/Nhạy) kèm theo thước đo phần trăm xác suất kháng Ciprofloxacin.
   * Vẽ biểu đồ lực lượng **SHAP đóng góp đặc trưng** cục bộ (Chart.js) cho thấy lý do tại sao mô hình đưa ra kết luận đó.
   * Tự động tạo **Báo cáo AI sinh học lâm sàng** và cung cấp cổng chat hỏi đáp về cơ chế kháng thuốc / gợi ý điều trị lâm sàng với Trợ lý AI.
3. **Chẩn Đoán Hàng Loạt (Batch Inference)**: Hỗ trợ kéo thả tệp CSV chứa dữ liệu của nhiều chủng, trả về bảng xem trước kết quả tức thì và nút tải xuống báo cáo kết quả tổng hợp đầy đủ dạng CSV.
4. **Lịch Sử Chẩn Đoán (History Logs)**: Lưu trữ tự động toàn bộ lịch sử phân tích vào SQLite, hỗ trợ tra cứu lại chi tiết kết quả hoặc xóa lịch sử.
5. **Giám Sát Dịch Tễ (Epidemiological Surveillance)**: Tổng hợp thống kê trực quan xu hướng đề kháng và phân bố tần suất xuất hiện các gen kháng thuốc điển hình trong tập mẫu nghiên cứu.
