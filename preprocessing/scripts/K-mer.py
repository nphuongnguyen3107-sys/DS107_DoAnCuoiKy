import os
import pandas as pd
from Bio import SeqIO
from sklearn.feature_extraction.text import CountVectorizer

# Đường dẫn tới thư mục chứa kết quả từ Bước 1
faa_dir = "faa_files"

sample_names = []
corpus = []

print("Bắt đầu đọc dữ liệu protein...")
# 1. Đọc tất cả các file .faa trong thư mục
for filename in os.listdir(faa_dir):
    if filename.endswith(".faa"):
        filepath = os.path.join(faa_dir, filename)
        
        # Lấy tên mẫu (bỏ đuôi .faa)
        sample_name = filename.replace(".faa", "")
        sample_names.append(sample_name)
        
        # Đọc file và nối tất cả các chuỗi protein của mẫu này lại với nhau
        aa_seqs = []
        for record in SeqIO.parse(filepath, "fasta"):
            aa_seqs.append(str(record.seq))
            
        # Các protein cách nhau bằng một khoảng trắng
        corpus.append(" ".join(aa_seqs))

print(f"Đã tải xong {len(corpus)} mẫu vi khuẩn.")

# 2. Đếm K-mer Axit Amin (Ví dụ: 3-mer)
# analyzer='char' báo cho máy tính biết đếm từng ký tự (axit amin)
# ngram_range=(3, 3) nghĩa là chỉ lấy các đoạn có độ dài đúng bằng 3
print("Đang trích xuất và đếm K-mer...")
vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 3))
X_kmer = vectorizer.fit_transform(corpus)

# 3. Chuyển kết quả thành Pandas DataFrame
kmer_df = pd.DataFrame(X_kmer.toarray(), columns=vectorizer.get_feature_names_out())

# Thêm cột tên mẫu vào đầu bảng để dễ dàng phân biệt
kmer_df.insert(0, "Sample", sample_names)

print("Hoàn thành! Kích thước ma trận K-mer:", kmer_df.shape)
# Lưu kết quả ra file CSV nếu muốn
kmer_df.to_csv("kmer_features.csv", index=False)