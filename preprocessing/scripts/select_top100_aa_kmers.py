"""
Select Top 100 Most Important AA K-mers and Train Hybrid Model
===============================================================
1. Load AA k-mer matrix (5027 genomes x 10695 features)
2. Feature selection: chi-square → top 100 k-mers
3. Combine with gene features (intersection of genomes)
4. Train and compare models
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("SELECT TOP 100 AA K-mers + Train Hybrid Model")
print("=" * 70)

# ===========================================================================
# STEP 1: Load data
# ===========================================================================
print("\n[1] Loading data...")

# Load AA k-mer features
df_aa = pd.read_csv('kmer_features.csv')
df_aa = df_aa.set_index('Sample')
print(f"   - AA k-mer matrix: {df_aa.shape}")

# Load gene features and labels
df_gene = pd.read_csv('X_train.csv', index_col=0)
df_y = pd.read_csv('y_train.csv')
y_full = df_y.set_index('Genome ID')['Resistant']

print(f"   - Gene features: {df_gene.shape}")
print(f"   - Labels: {len(y_full)}")

# ===========================================================================
# STEP 2: Align genomes
# ===========================================================================
print("\n[2] Aligning genomes...")

# Get common genomes
common_genomes = df_aa.index.intersection(df_gene.index).intersection(y_full.index)
print(f"   - Common genomes: {len(common_genomes)}")

# Subset to common genomes
df_aa_aligned = df_aa.loc[common_genomes]
df_gene_aligned = df_gene.loc[common_genomes]
y_aligned = y_full.loc[common_genomes]

print(f"   - AA k-mers: {df_aa_aligned.shape}")
print(f"   - Gene features: {df_gene_aligned.shape}")

# ===========================================================================
# STEP 3: Feature selection - Top 100 AA k-mers
# ===========================================================================
print("\n[3] Selecting top 100 AA k-mers by chi-square...")

X_aa = df_aa_aligned.values
y_vec = y_aligned.values

# Chi-square selection
chi_selector = SelectKBest(score_func=chi2, k=100)
chi_selector.fit(X_aa, y_vec)

# Get selected k-mer names
selected_mask = chi_selector.get_support()
selected_kmers = df_aa_aligned.columns[selected_mask]

print(f"   - Selected top 100 AA k-mers")

# Chi-square scores for interpretation
kmer_scores = pd.DataFrame({
    'kmer': selected_kmers,
    'chi2_score': chi_selector.scores_[selected_mask]
}).sort_values('chi2_score', ascending=False)

print("\n   Top 20 discriminatory AA k-mers:")
print(kmer_scores.head(20).to_string(index=False))

# Subset to top 100 k-mers
df_aa_top100 = df_aa_aligned[selected_kmers]
print(f"\n   - AA k-mer matrix after selection: {df_aa_top100.shape}")

# ===========================================================================
# STEP 4: Combine with gene features
# ===========================================================================
print("\n[4] Combining AA k-mers + gene features...")

# Select top 50 gene features by variance (to avoid too many features)
gene_cols = [col for col in df_gene_aligned.columns if col != 'Genome ID']
gene_variance = df_gene_aligned[gene_cols].var().sort_values(ascending=False)
top_gene_cols = gene_variance.head(50).index.tolist()
df_gene_top50 = df_gene_aligned[top_gene_cols]

# Combine: 100 AA k-mers + 50 gene features = 150 features
df_combined = pd.concat([df_aa_top100, df_gene_top50], axis=1)
print(f"   - Combined features: {df_combined.shape[1]}")
print(f"   - Samples: {df_combined.shape[0]}")

# ===========================================================================
# STEP 5: Train models
# ===========================================================================
print("\n[5] Training models...")

X = df_combined.values
y = y_aligned.values

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight='balanced')
}

results = {}

for name, model in models.items():
    print(f"\n   --- {name} ---")

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_proba)
    }

    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
    metrics['CV ROC-AUC'] = f"{cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})"

    results[name] = metrics

    print(f"   Accuracy:  {metrics['Accuracy']:.4f}")
    print(f"   Precision: {metrics['Precision']:.4f}")
    print(f"   Recall:    {metrics['Recall']:.4f}")
    print(f"   F1:        {metrics['F1']:.4f}")
    print(f"   ROC-AUC:   {metrics['ROC-AUC']:.4f}")
    print(f"   CV ROC-AUC: {metrics['CV ROC-AUC']}")

# ===========================================================================
# STEP 6: Compare with baselines
# ===========================================================================
print("\n" + "=" * 70)
print("COMPARISON: AA k-mer (100) vs Gene-only vs Hybrid (NT+Gene)")
print("=" * 70)

# Baseline scores from previous runs
baseline = {
    'Model': ['Gene-only (LR)', 'Hybrid NT+Gene (LR)', 'AA 100+Gene (LR)'],
    'ROC-AUC': [0.8862, 0.8866, results['Logistic Regression']['ROC-AUC']],
    'Recall': [0.6122, 0.7653, results['Logistic Regression']['Recall']],
    'Precision': [0.9160, 0.7895, results['Logistic Regression']['Precision']],
    'F1': [0.7339, 0.7772, results['Logistic Regression']['F1']]
}

comparison = pd.DataFrame(baseline)
print(comparison.to_string(index=False))

# ===========================================================================
# STEP 7: Feature importance
# ===========================================================================
print("\n[6] Top features by importance...")

rf_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight='balanced')
rf_model.fit(X_train_scaled, y_train)

importances = rf_model.feature_importances_
feature_names = list(df_combined.columns)

feature_imp = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print("\n   Top 30 Most Important Features:")
print(feature_imp.head(30).to_string(index=False))

# ===========================================================================
# STEP 8: Save results
# ===========================================================================
print("\n[7] Saving results...")

# Top 100 k-mers
kmer_scores.to_csv('top_100_aa_kmers.csv', index=False)

# Comparison
comparison.to_csv('aa_vs_gene_comparison.csv', index=False)

# Feature importance
feature_imp.to_csv('hybrid_aa_gene_importance.csv', index=False)

# Combined features
df_combined.to_csv('hybrid_aa_gene_features.csv')

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Best model: Logistic Regression")
print(f"ROC-AUC: {results['Logistic Regression']['ROC-AUC']:.4f}")
print(f"Recall: {results['Logistic Regression']['Recall']:.4f}")
print(f"Precision: {results['Logistic Regression']['Precision']:.4f}")
print(f"\nFiles saved:")
print(f"  - top_100_aa_kmers.csv (100 most important AA k-mers)")
print(f"  - aa_vs_gene_comparison.csv (performance comparison)")
print(f"  - hybrid_aa_gene_importance.csv (feature importance)")
print(f"  - hybrid_aa_gene_features.csv (combined feature matrix)")