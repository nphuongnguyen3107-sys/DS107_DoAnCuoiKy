import os, subprocess
import pandas as pd
from tqdm import tqdm

fasta_dir = 'fasta_files' 
amr_dir = 'amr_results'   
os.makedirs(amr_dir, exist_ok=True)

fasta_files = [f for f in os.listdir(fasta_dir) if f.endswith('.fasta')]
print(f"--- BẮT ĐẦU QUÉT {len(fasta_files)} VI KHUẨN ---")

for filename in tqdm(fasta_files, desc="Tiến độ quét gen"):
    basename = filename.replace('.fasta', '')
    input_path = os.path.join(fasta_dir, filename)
    output_path = os.path.join(amr_dir, f"{basename}.tsv")
    
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        cmd = f"amrfinder -n {input_path} -O Escherichia > {output_path}"
        try:
            subprocess.run(cmd, shell=True, check=True, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            pass 

print("\n--- ĐANG GỘP KẾT QUẢ THÀNH MA TRẬN ---")
amr_data = {}
tsv_files = [f for f in os.listdir(amr_dir) if f.endswith('.tsv')]

for filename in tqdm(tsv_files, desc="Đang gộp file TSV"):
    genome_id = filename.replace('.tsv', '')
    file_path = os.path.join(amr_dir, filename)
    genes_found = {}
    if os.path.getsize(file_path) > 0:
        try:
            df_amr = pd.read_csv(file_path, sep='\t')
            if 'Element symbol' in df_amr.columns:
                for gene in df_amr['Element symbol'].dropna():
                    genes_found[gene] = 1 
        except Exception:
            pass 
    amr_data[genome_id] = genes_found

df_X_amr = pd.DataFrame.from_dict(amr_data, orient='index').fillna(0).astype('int8')
df_X_amr.to_csv('X_features_AMR.csv', index_label='Genome ID')
print("Xong Pha 2.1! Đã lưu X_features_AMR.csv")