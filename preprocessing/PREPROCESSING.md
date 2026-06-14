# PHẦN TIỀN XỬ LÝ - BÁO CÁO KỸ THUẬT

## 1. DỮ LIỆU ĐẦU VÀO

### 1.1. Dữ liệu Genotype (FASTA Files)

**Nguồn**: BVR-BRC (Bacterial and Viral Bioinformatics Resource Center)

**Định dạng**: Mỗi file chứa một genome E. coli hoàn chỉnh

**Cấu trúc file mẫu**:
```fasta
>accn|562.100003.con.0029   ERR4037446_contig_29   [Escherichia coli c1b4da3e-7bb9-11e9-a8d3-68b59976a384 | 562.100003]
gcaaaaagcctgcagtggacgcacaacgatgaatgccttaaaccggggcaattagacggg
atggaagtcgagaatgatttaagccagtcggctttgctgctgacagtgccacaggcttac
...
```

**Thống kê ban đầu**:
- Tổng số files: **7,334 genomes**
- Mỗi genome: 50-200 contigs (ước tính)
- Định dạng nucleotide: DNA (A, T, G, C, N)

---

### 1.2. Dữ liệu Phenotype (Metadata)

**File**: `BVBRC_genome_amr.csv`

**Cấu trúc columns**:

| Column | Mô tả |
|--------|-------|
| `Taxon ID` | ID taxonom (562 = E. coli) |
| `Genome ID` | Định danh genome (ví dụ: `562.100003`) |
| `Genome Name` | Tên đầy đủ của isolate |
| `Antibiotic` | Tên kháng sinh (chỉ dùng ciprofloxacin) |
| `Resistant Phenotype` | Resistant / Susceptible / Intermediate |
| `Laboratory Typing Method` | Phương pháp xác định (Disk diffusion, MIC, etc.) |
| `Testing Standard` | Chuẩn (EUCAST, CLSI) |

**Thống kê ban đầu**:
- Tổng rows: 8,635
- Unique Genome IDs: 7,334
- Antibiotics: Nhiều loại → chỉ chọn **ciprofloxacin**

---

## 2. QUY TRÌNH TIỀN XỬ LÝ CHI TIẾT

### **Bước 1: Làm sạch & Lọc Dữ liệu**

#### 1.1. Lọc FASTA files theo CSV

**Mục đích**: Chỉ giữ lại genomes có trong metadata

```python
# Đọc danh sách Genome IDs từ CSV
import pandas as pd

df_metadata = pd.read_csv('BVBRC_genome_amr.csv')
genome_ids_in_csv = set(df_metadata['Genome ID'].unique())

# Lọc FASTA files
import os
fasta_dir = 'fasta_files'
fasta_files = [f for f in os.listdir(fasta_dir) if f.endswith('.fasta')]

# Kiểm tra và giữ lại files có tên trong CSV
valid_fasta = []
for fasta in fasta_files:
    genome_id = fasta.replace('.fasta', '')
    if genome_id in genome_ids_in_csv:
        valid_fasta.append(fasta)

# Kết quả: 5,027 genomes
```

**Kết quả**:
- Trước: 7,334 files
- Sau: **5,027 files** (có trong CSV)

---

#### 1.2. Lọc & Mã hóa Metadata

**Tách theo kháng sinh**:
```python
# Chỉ giữ ciprofloxacin
df_cipro = df_metadata[df_metadata['Antibiotic'] == 'ciprofloxacin']
# Rows: 6,104
```

**Loại bỏ nhãn không xác định**:
```python
# Chỉ giữ Resistant và Susceptible
df_clean = df_cipro[
    df_cipro['Resistant Phenotype'].isin(['Resistant', 'Susceptible'])
]
```

**Mã hóa nhị phân**:
```python
label_map = {
    'Resistant': 1,
    'Susceptible': 0
}
df_clean['label'] = df_clean['Resistant Phenotype'].map(label_map)
```

**Aggregate by Genome** (nếu một genome có nhiều entries):
```python
# Lấy label max (nếu có conflict, ưu tiên Resistant)
df_labels = df_clean.groupby('Genome ID')['label'].max().reset_index()
```

**Kết quả**:
- Rows sau groupby: **5,027 genomes**
- Resistant (1): ~2,000
- Susceptible (0): ~3,000

---

### **Bước 2: Trích xuất Đặc trưng Gen (AMR Gene Detection)**

#### 2.1. Công cụ: AMRFinderPlus

**Lý do chọn**: 
- Tool chuẩn của NCBI cho bacterial AMR detection
- Database cập nhật: CARD + ARDB + custom HMMs
- Phát hiện cả gene và point mutations

**Cài đặt**: Tải từ NCBI website

**Command line**:
```bash
#!/bin/bash
fasta_dir="fasta_files"
output_dir="amr_results"
mkdir -p $output_dir

for fasta in $fasta_dir/*.fasta; do
    basename=$(basename $fasta .fasta)
    output="$output_dir/$basename.tsv"

    # Chỉ chạy nếu chưa có output hoặc file rỗng
    if [ ! -f "$output" ] || [ ! -s "$output" ]; then
        amrfinder -n $fasta -O Escherichia > $output 2>/dev/null
    fi
done
```

**Output format (TSV)**:
```
Protein id    Contig id    Start    Stop    Strand    Element symbol    Element name    Scope    Type    Subtype    Class    Subclass    Method    Target length    Reference sequence length    % Coverage    % Identity
NA    accn|562.100003.con.0002    115637    117526    -    parE_I529L    Escherichia quinolone resistant ParE    core    AMR    POINT    QUINOLONE    QUINOLONE    POINTX    630    630    100.00    99.52
NA    accn|562.100003.con.0003    45678    46789    +    gyrA_D87N    Escherichia quinolone resistant gyrA    core    AMR    POINT    QUINOLONE    QUINOLONE    POINTX    350    350    100.00    100.00
```

---

#### 2.2. Tạo Gene Presence/Absence Matrix

**Thu thập tất cả gene symbols**:
```python
import pandas as pd
from collections import defaultdict

all_genes = set()
genome_genes = {}

for tsv_file in os.listdir('amr_results'):
    if tsv_file.endswith('.tsv'):
        genome_id = tsv_file.replace('.tsv', '')
        file_path = os.path.join('amr_results', tsv_file)

        if os.path.getsize(file_path) > 0:
            df = pd.read_csv(file_path, sep='\t')
            if 'Element symbol' in df.columns:
                genes = set(df['Element symbol'].dropna())
                genome_genes[genome_id] = genes
                all_genes.update(genes)

print(f"Unique AMR genes: {len(all_genes)}")
# Output: ~400+ genes
```

**Lọc gene quá hiếm** (optional):
```python
# Chỉ giữ gene xuất hiện trong >= 1% genomes
gene_prevalence = {}
for gene in all_genes:
    count = sum(1 for genes in genome_genes.values() if gene in genes)
    gene_prevalence[gene] = count / len(genome_genes)

common_genes = [g for g, p in gene_prevalence.items() if p >= 0.01]
print(f"Common genes (>=1%): {len(common_genes)}")
# Output: 210 genes
```

**Tạo ma trận nhị phân**:
```python
# Sắp xếp gene list
gene_list = sorted(common_genes)

# Tạo DataFrame
data = []
for genome_id in sorted(genome_genes.keys()):
    row = {}
    for gene in gene_list:
        row[gene] = 1 if gene in genome_genes.get(genome_id, set()) else 0
    data.append(row)

df_X_amr = pd.DataFrame(data, index=sorted(genome_genes.keys()))
df_X_amr.to_csv('X_features_AMR.csv')
```

**Kết quả**:
- Shape: **5,027 genomes** × **210 genes**
- Dữ liệu: Binary (0/1)
- Sparsity: ~98% zeros

---

### **Bước 3: Trích xuất AA K-mer Features**

#### 3.1. Dịch mã DNA → Protein

**Sử dụng Prodigal** (tool chuẩn cho gene prediction):

```bash
# Cài đặt Prodigal
# https://github.com/hyattpd/Prodigal

# Dịch toàn bộ FASTA files
for fasta in fasta_files/*.fasta; do
    basename=$(basename $fasta .fasta)
    prodigal -i $fasta -a proteins/$basename.faa -p meta
done
```

**Hoặc dùng Python custom translator**:
```python
CODON_TABLE = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
    'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'*', 'TAG':'*',
    'TGC':'C', 'TGT':'C', 'TGA':'*', 'TGG':'W',
}

def translate_dna_to_protein(dna_sequence):
    """Translate DNA sequence to protein."""
    dna_sequence = dna_sequence.upper().replace('N', '')
    protein = []
    for i in range(0, len(dna_sequence) - 2, 3):
        codon = dna_sequence[i:i+3]
        if len(codon) == 3:
            aa = CODON_TABLE.get(codon, 'X')  # X = unknown
            protein.append(aa)
    return ''.join(protein)
```

---

#### 3.2. Trích xuất K-mer

**Hàm trích xuất k-mer từ protein sequence**:
```python
from collections import Counter

def count_kmers(sequence, k=3):
    """Count k-mer frequencies in a sequence."""
    kmers = Counter()
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        if 'X' not in kmer and '*' not in kmer:  # Skip unknown/stop
            kmers[kmer] += 1
    return kmers

def get_kmer_frequencies_from_fasta(fasta_file, k=3):
    """Extract normalized k-mer frequencies from FASTA."""
    total_kmers = Counter()

    with open(fasta_file, 'r') as f:
        current_seq = []
        for line in f:
            if line.startswith('>'):
                # Process previous sequence
                if current_seq:
                    seq = ''.join(current_seq)
                    protein = translate_dna_to_protein(seq)
                    total_kmers.update(count_kmers(protein, k))
                current_seq = []
            else:
                current_seq.append(line.strip())

        # Process last sequence
        if current_seq:
            seq = ''.join(current_seq)
            protein = translate_dna_to_protein(seq)
            total_kmers.update(count_kmers(protein, k))

    # Normalize to frequencies
    total_count = sum(total_kmers.values())
    if total_count == 0:
        return {}

    return {kmer: count/total_count for kmer, count in total_kmers.items()}
```

**Xử lý tất cả genomes**:
```python
import os

fasta_dir = 'fasta_files'
genome_to_fasta = {}

# Map genome IDs to FASTA files
for fasta in os.listdir(fasta_dir):
    if fasta.endswith('.fasta'):
        gid = fasta.replace('.fasta', '')
        genome_to_fasta[gid] = os.path.join(fasta_dir, fasta)

# Extract k-mers for all genomes
all_kmers = set()
kmer_data = {}

for i, (gid, fasta_path) in enumerate(genome_to_fasta.items()):
    kmers = get_kmer_frequencies_from_fasta(fasta_path, k=3)
    kmer_data[gid] = kmers
    all_kmers.update(kmers.keys())

    if (i+1) % 500 == 0:
        print(f"Processed {i+1}/{len(genome_to_fasta)}")

print(f"Total unique 3-mers: {len(all_kmers)}")
# Output: ~10,695 unique AA 3-mers
```

---

#### 3.3. Tạo K-mer Matrix & Lọc

**Tạo DataFrame**:
```python
import pandas as pd

# Convert to DataFrame
genome_ids = sorted(kmer_data.keys())
kmer_cols = sorted(all_kmers)

df_kmer = pd.DataFrame(0.0, index=genome_ids, columns=kmer_cols)

for gid, kmers in kmer_data.items():
    for kmer, freq in kmers.items():
        df_kmer.loc[gid, kmer] = freq

print(f"K-mer matrix shape: {df_kmer.shape}")
# Output: (5027, 10695)
```

**Lọc rare k-mers**:
```python
# Remove k-mers present in < 0.1% of genomes
prevalence = (df_kmer > 0).mean()
rare_kmers = prevalence[prevalence < 0.001].index
df_kmer_filtered = df_kmer.drop(columns=rare_kmers)

print(f"After filtering rare k-mers: {df_kmer_filtered.shape[1]}")
# Output: ~637 features
```

---

### **Bước 4: Feature Selection**

**So sánh 3 phương pháp**:

#### 4.1. Chi-Square Test
```python
from sklearn.feature_selection import SelectKBest, chi2

selector_chi2 = SelectKBest(score_func=chi2, k=100)
selector_chi2.fit(df_kmer_filtered, y_aligned)

chi2_features = df_kmer_filtered.columns[selector_chi2.get_support()]
chi2_scores = selector_chi2.scores_[selector_chi2.get_support()]
```

#### 4.2. ANOVA F-test
```python
from sklearn.feature_selection import f_classif

selector_anova = SelectKBest(score_func=f_classif, k=100)
selector_anova.fit(df_kmer_filtered, y_aligned)

anova_features = df_kmer_filtered.columns[selector_anova.get_support()]
anova_scores = selector_anova.scores_[selector_anova.get_support()]
```

#### 4.3. Random Forest Importance
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)
rf.fit(df_kmer_filtered, y_aligned)

importances = rf.feature_importances_
top_idx = np.argsort(importances)[::-1][:100]
rf_features = df_kmer_filtered.columns[top_idx]
rf_scores = importances[top_idx]
```

**Kết quả so sánh**:

| Method | ROC-AUC | Recall | Precision | F1 |
|--------|---------|--------|-----------|----|
| Chi-Square | 0.9096 | 80.6% | 79.0% | 79.8% |
| ANOVA | 0.9054 | 82.1% | 78.5% | 80.3% |
| **RF Importance** | **0.9133** | 81.6% | **80.8%** | **81.2%** |

→ **Chọn RF Importance** cho final model.

---

### **Bước 5: Tích hợp Đặc trưng**

#### 5.1. Align genomes

```python
# Load tất cả nguồn dữ liệu
df_gene = pd.read_csv('X_features_AMR.csv', index_col=0)  # 210 genes
df_kmer_selected = df_kmer_filtered[rf_features]           # 100 k-mers
df_labels = pd.read_csv('y_train.csv', index_col='Genome ID')

# Get common genomes
common_ids = df_gene.index.intersection(
    df_kmer_selected.index
).intersection(df_labels.index)

print(f"Common genomes: {len(common_ids)}")
# Output: 2,404
```

#### 5.2. Kết hợp features

```python
# Subset to common genomes
X_gene = df_gene.loc[common_ids]
X_kmer = df_kmer_selected.loc[common_ids]
y = df_labels.loc[common_ids, 'Resistant']

# Combine
X_combined = pd.concat([X_gene, X_kmer], axis=1)

print(f"Final feature matrix: {X_combined.shape}")
# Output: (2404, 310)
print(f"  - Gene features: {X_gene.shape[1]}")
print(f"  - AA k-mer features: {X_kmer.shape[1]}")
```

---

#### 5.3. Chuẩn hóa

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)

# Lưu scaler để inference
import pickle
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
```

---

### **Bước 6: Lưu Dữ liệu Cuối cùng**

**Tệp dữ liệu**:

| File | Nội dung | Shape |
|------|----------|-------|
| `final_combined_features.csv` | Feature matrix (X) | 2404 × 310 |
| `y_final.csv` | Target labels (y) | 2404 × 2 |
| `scaler.pkl` | StandardScaler đã fit | - |
| `final_rf_model.pkl` | Model đã train | - |
| `rf_top_100_kmers_final.csv` | Top 100 AA k-mers | 100 × 2 |

**Code lưu files**:
```python
# Save feature matrix
X_combined.to_csv('final_combined_features.csv')

# Save labels
y_df = pd.DataFrame({
    'Genome ID': y.index,
    'Resistant': y.values
})
y_df.to_csv('y_final.csv', index=False)

# Save feature names
feature_names = list(X_combined.columns)
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)
```

---

## 3. KẾT QUẢ TIỀN XỬ LÝ TỔNG HỢP

### 3.1. Thống kê Dữ liệu

| Stage | Số lượng | Ghi chú |
|-------|----------|---------|
| **FASTA files ban đầu** | 7,334 | Từ BV-BRC |
| **FASTA sau lọc** | 5,027 | Chỉ giữ genomes có trong CSV |
| **Genomes có phenotype** | 5,027 | Sau clean metadata |
| **Genomes có đủ features** | **2,404** | Intersection của gene + k-mer + labels |
| **Gene features** | 210 | Từ AMRFinderPlus |
| **AA k-mer features** | 100 | Top RF-selected |
| **Total features** | **310** | Final matrix |

### 3.2. Phân bổ Labels

```
Resistant (1):  981 genomes (40.8%)
Susceptible (0): 1,423 genomes (59.2%)
Total:          2,404 genomes
Imbalance ratio: 1.45:1
```

### 3.3. Tính chất Ma trận

| Metric | Gene features (210) | AA k-mer (100) | Combined (310) |
|--------|---------------------|----------------|----------------|
| Sparsity | ~98% | ~99.9% | ~98.5% |
| Mean per genome | 3.0 genes | ~50 k-mers | ~3.5 active features |
| Value range | 0/1 | [0, 0.05] | Mixed |

---

## 4. LƯU Ý KỸ THUẬT

### 4.1. Memory Requirements

- **Raw FASTA**: ~50 GB (7,334 files)
- **AMR results**: ~500 MB (TSV files)
- **K-mer matrix (full)**: 5,027 × 10,695 ≈ 5 GB (float64)
- **Final matrix**: 2,404 × 310 ≈ 6 MB (float64)

### 4.2. Computation Time

| Step | Thời gian ước tính |
|------|-------------------|
| AMRFinderPlus scan | 2-4 hours (7,334 files) |
| AA k-mer extraction | 6-12 hours (Python) |
| Feature selection | 30 minutes |
| Model training | 5-10 minutes |

### 4.3. Reproducibility

**Random seeds**:
```python
RANDOM_STATE = 42  # Dùng xuyên suốt
```

**Python versions**:
```
Python: 3.14
pandas: 2.x
scikit-learn: 1.5+
numpy: 2.x
```

**External tools**:
```
AMRFinderPlus: v3.12+
Prodigal: v2.6.3+ (optional)
```

---

## 5. CODE IMPLEMENTATION

**Các script chính đã triển khai**:

| Script | Mục đích | Thời gian chạy |
|--------|----------|----------------|
| `merge_dataset.py` | Tạo X_train.csv, y_train.csv | ~5 phút |
| `select_top100_aa_kmers.py` | Chọn top 100 k-mers + train | ~30 phút |
| `feature_selection_simple.py` | So sánh 3 phương pháp | ~20 phút |
| `final_rf_model.py` | Train final model | ~10 phút |

---

## 6. KIỂM TRA CHẤT LƯỢNG

### 6.1. Data Integrity Checks

```python
# Kiểm tra không có NaN
assert not X_combined.isna().any().any(), "NaN values found!"

# Kiểm tra labels trong [0, 1]
assert y.isin([0, 1]).all(), "Invalid labels!"

# Kiểm tra shapes match
assert len(X_combined) == len(y), "X and y length mismatch!"
```

### 6.2. Feature Validation

```python
# Kiểm tra gene features là binary
gene_cols = [c for c in X_combined.columns if c in GENE_NAMES]
assert (X_combined[gene_cols].isin([0, 1]).all().all()), "Gene features not binary!"

# Kiểm tra k-mer features là float trong [0, 1]
kmer_cols = [c for c in X_combined.columns if c not in GENE_NAMES]
assert (X_combined[kmer_cols] >= 0).all().all(), "Negative k-mer frequencies!"
assert (X_combined[kmer_cols] <= 1).all().all(), "K-mer freq > 1!"
```

---

**Kết thúc phần Tiền xử lý.** Dữ liệu đã sẵn sàng cho huấn luyện model.
