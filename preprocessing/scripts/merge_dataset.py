import pandas as pd
import numpy as np

print("=" * 60)
print("STEP 1: Merge AMR Features with Phenotype Labels")
print("=" * 60)

# Load AMR features matrix
print("\n1. Loading AMR features matrix...")
df_X = pd.read_csv('X_features_AMR.csv', index_col=0)
print(f"   - Shape: {df_X.shape[0]} genomes x {df_X.shape[1]} features")
print(f"   - Feature columns: {df_X.shape[1]}")

# Load phenotype data
print("\n2. Loading phenotype data...")
df_pheno = pd.read_csv('BVBRC_genome_amr.csv')
print(f"   - Total rows: {len(df_pheno)}")
print(f"   - Unique genomes: {df_pheno['Genome ID'].nunique()}")
print(f"   - Antibiotics: {df_pheno['Antibiotic'].unique()}")

# Convert phenotype to binary labels
print("\n3. Converting phenotype to binary labels...")
# Keep only ciprofloxacin data for now
df_cipro = df_pheno[df_pheno['Antibiotic'] == 'ciprofloxacin'].copy()
df_cipro['Resistant'] = df_cipro['Resistant Phenotype'].apply(
    lambda x: 1 if x == 'Resistant' else 0
)
print(f"   - Ciprofloxacin rows: {len(df_cipro)}")

# Aggregate by genome (if multiple entries per genome)
df_labels = df_cipro.groupby('Genome ID')['Resistant'].max().reset_index()
print(f"   - Unique genomes with labels: {len(df_labels)}")

# Merge features with labels
print("\n4. Merging features with labels...")
df_merged = df_X.reset_index().merge(df_labels, on='Genome ID', how='inner')
print(f"   - Merged shape: {df_merged.shape}")

# Separate X and y
feature_cols = [col for col in df_merged.columns if col not in ['Genome ID', 'Resistant']]
X = df_merged[['Genome ID'] + feature_cols]
y = df_merged[['Genome ID', 'Resistant']]

# Save results
print("\n5. Saving results...")
X.to_csv('X_train.csv', index=False)
y.to_csv('y_train.csv', index=False)

# Summary statistics
print("\n" + "=" * 60)
print("SUMMARY - STEP 1 COMPLETE")
print("=" * 60)
print(f"Total samples: {len(X)}")
print(f"Features: {len(feature_cols)}")
print(f"\nLabel distribution:")
print(f"  - Resistant (1): {y['Resistant'].sum()}")
print(f"  - Susceptible (0): {len(y) - y['Resistant'].sum()}")
print(f"\nFiles created:")
print(f"  - X_train.csv ({len(X)} rows x {len(feature_cols)+1} columns)")
print(f"  - y_train.csv ({len(y)} rows x 2 columns)")