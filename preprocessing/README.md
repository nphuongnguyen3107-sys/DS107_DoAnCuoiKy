# AMR Data Pipeline

## Giới thiệu

Pipeline tiền xử lý dữ liệu genome E. coli để tạo feature matrix cho bài toán dự đoán khả năng kháng ciprofloxacin.

**Kết quả đầu ra chính:**
- `data/processed/final_rf_combined_features.csv`: Ma trận đặc trưng cuối cùng (210 gene features + 100 AA k-mers) đã được chuẩn hóa và sẵn sàng cho huấn luyện mô hình.

---

## Yêu cầu Hệ thống

### Python Dependencies

```bash
pip install -r requirements.txt
```

### AMR Gene Detection (chỉ khi chạy từ đầu với FASTA thô)

```bash
# Cần cài đặt AMRFinderPlus từ NCBI
# https://www.ncbi.nlm.nih.gov/pathogens/antimicrobial-resistance/AMRFinderPlus/
```

Lưu ý: AMRFinderPlus chỉ cần thiết nếu bạn muốn chạy lại toàn bộ pipeline từ dữ liệu FASTA thô. Nếu chỉ cần tái tạo từ dữ liệu trung gian, bước này có thể bỏ qua.

---

## Cấu trúc Files

```
submission_package/
├── scripts/                    # Source code cho pipeline
│   ├── run_amr.py              # Bước 1: Chạy AMRFinderPlus, tạo gene features
│   ├── merge_dataset.py        # Bước 2: Gộp gene features + phenotype labels
│   ├── K-mer.py                # Bước 3: Trích xuất amino acid k-mer features
│   └── select_top100_aa_kmers.py  # Bước 4: Chọn top 100 k-mers + kết hợp cuối cùng
├── data/
│   ├── raw/
│   │   └── BVBRC_genome_amr.csv   # Metadata phenotype (Resistant/Susceptible)
│   └── processed/
│       ├── X_features_AMR.csv       # Ma trận gene features (output từ run_amr.py)
│       ├── X_train.csv             # Gene features đã gộp labels
│       ├── y_train.csv             # Nhãn kháng thuốc
│       ├── kmer_features.csv       # Ma trận AA k-mer đầy đủ (5027 x 10695)
│       └── final_rf_combined_features.csv  # Output cuối: 2404 x 310
├── PREPROCESSING.md            # Tài liệu kỹ thuật chi tiết
└── requirements.txt            # Danh sách thư viện Python
```

---

## Hướng dẫn Chạy Pipeline

Pipeline chạy theo thứ tự:

```bash
# Bước 1: Tạo AMR gene features từ FASTA files
# Cần đặt FASTA files vào data/raw/fasta_files/ và cài AMRFinderPlus
python scripts/run_amr.py
# Output: data/processed/X_features_AMR.csv

# Bước 2: Gộp gene features với phenotype labels
python scripts/merge_dataset.py
# Output: data/processed/X_train.csv, data/processed/y_train.csv

# Bước 3: Trích xuất amino acid k-mer features
# Cần đặt protein FASTA (.faa) vào data/raw/faa_files/
python scripts/K-mer.py
# Output: data/processed/kmer_features.csv

# Bước 4: Chọn top 100 AA k-mers và kết hợp với gene features
python scripts/select_top100_aa_kmers.py
# Output: data/processed/final_rf_combined_features.csv
```

Sau bước 4, file `final_rf_combined_features.csv` sẽ chứa ma trận đặc trưng cuối cùng (2404 samples x 310 features) sẵn sàng cho bước huấn luyện mô hình.

---

## Kết quả Pipeline

| File | Mô tả | Kích thước |
|------|-------|------------|
| `X_features_AMR.csv` | Ma trận gene presence/absence | 5027 x ~210 |
| `X_train.csv` | Gene features đã align với labels | ~2404 x 210 |
| `y_train.csv` | Nhãn nhị phân (Resistant=1, Susceptible=0) | ~2404 rows |
| `kmer_features.csv` | Ma trận AA k-mer đầy đủ | 5027 x 10695 |
| `final_rf_combined_features.csv` | **Output cuối** - kết hợp gene + top 100 k-mers | **2404 x 310** |

---

## Phương pháp Feature Selection

Script `select_top100_aa_kmers.py` thực hiện:

1. Load AA k-mer matrix đầy đủ
2. Align genomes với gene features và labels
3. Chọn top 100 k-mers bằng Chi-square test
4. Kết hợp với toàn bộ 210 gene features
5. Xuất ma trận cuối cùng

---

## Lưu ý quan trọng

### Về dữ liệu đầu vào

Do giới hạn kích thước nộp bài, các dữ liệu raw sau **không được bao gồm** trong package:

- `data/raw/fasta_files/` (genome FASTA, ~10 GB)
- `data/raw/faa_files/` (protein FASTA, ~10 GB)
- `data/raw/amr_results/` (kết quả AMRFinderPlus, ~50 MB)

Tuy nhiên, pipeline đã bao gồm:
- File phenotype `BVBRC_genome_amr.csv`
- Các ma trận đặc trưng đã xử lý sẵn trong `data/processed/`

Nếu muốn chạy lại từ đầu với FASTA gốc, cần:
1. Cài đặt AMRFinderPlus
2. Đặt FASTA files vào `data/raw/fasta_files/`
3. Chạy `scripts/run_amr.py`

### Về môi trường chạy

- Pipeline Python có thể chạy trên Windows nếu đã cài đủ thư viện.
- AMRFinderPlus và Prodigal (nếu dùng) cần chạy trên Ubuntu/WSL2.

---

## References

1. **AMRFinderPlus:** NCBI AMR gene detection tool
2. **Dataset:** BVBRC E. coli AMR database
3. **Pipeline design:** Based on best practices for AMR prediction from genomic data
