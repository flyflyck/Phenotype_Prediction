# coding: utf-8

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.random_projection import SparseRandomProjection
from sklearn.utils import resample

from sklearn.linear_model import (
    LogisticRegression,
    LogisticRegressionCV,
    PassiveAggressiveClassifier,
    Ridge
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    mean_squared_error,
    confusion_matrix,
    precision_score,
    recall_score,
    auc,
    roc_curve
)

import pgenlib

# Paths
root_data_path = '/home/amele'
root_fname = 'gbe_ukb_top3phenotypes_10Ksample'
pgen_path = f'{root_data_path}/{root_fname}.pgen'.encode()
pvar_file = f'{root_data_path}/{root_fname}.pvar'
psam_file = f'{root_data_path}/{root_fname}.psam'
tsv_file = f'{root_data_path}/{root_fname}_phenotypes.tsv'

# Read genotype matrix
pgen_reader = pgenlib.PgenReader(pgen_path)
n_variants = pgen_reader.get_variant_ct()
n_samples = pgen_reader.get_raw_sample_ct()
calldata = np.empty((n_variants, 2 * n_samples), dtype=np.int32, order='C')
pgen_reader.read_alleles_range(0, n_variants, calldata)
data = calldata.T

# Load phenotype table
df_tsv = pd.read_csv(tsv_file, sep='\t')

# -------------------------------------------------------------------
# Compare runtime scaling of dimension-reduction methods
# -------------------------------------------------------------------
def data_size_scaling(estimator, data, sizes=[100, 1000, 5000, 9914], n_runs=5):
    records = []
    for size in sizes:
        for _ in range(n_runs):
            subsample = resample(data, n_samples=size, random_state=123)
            start = time.time()
            estimator.fit(subsample)
            runtime = time.time() - start
            records.append((size, runtime))
    return pd.DataFrame(records, columns=('dataset size', 'runtime (s)'))

all_algorithms = [
    PCA(),
    SparseRandomProjection()
]

performance_data = {}
for algo in all_algorithms:
    name = algo.__class__.__name__
    performance_data[name] = data_size_scaling(algo, data)
    print(f"[{time.asctime()}] Completed scaling for {name}")

plt.figure(figsize=(8, 6))
for name, df_perf in performance_data.items():
    sns.regplot(
        x='dataset size',
        y='runtime (s)',
        data=df_perf,
        order=2,
        label=name
    )
plt.legend()
plt.title('Runtime vs Dataset Size (Dimension Reduction)')
plt.xlabel('Dataset Size')
plt.ylabel('Runtime (s)')
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------
# Model comparison for three binary phenotypes
# -------------------------------------------------------------------
phenotypes = ['BIN_FC2001747', 'INI30790', 'INI30100']
models = [
    ('LR', LogisticRegression()),
    ('LDA', LinearDiscriminantAnalysis()),
    ('KNN', KNeighborsClassifier()),
    ('CART', DecisionTreeClassifier()),
    ('NB', GaussianNB()),
    ('SVM', SVC())
]
seed = 42

for pheno in phenotypes:
    y = df_tsv[pheno]
    X_train, X_test, y_train, y_test = train_test_split(
        data, y, test_size=0.2, random_state=seed
    )
    print(f"\n=== Results for phenotype {pheno} ===")
    for name, model in models:
        cv = KFold(n_splits=10, shuffle=True, random_state=seed)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        print(f"{name}: CV accuracy {scores.mean():.3f} ± {scores.std():.3f}")

    # final evaluation on test set
    compare = []
    for name, model in models:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        fp, tp, _ = roc_curve(y_test, pred)
        compare.append({
            'Model': name,
            'Train Acc': model.score(X_train, y_train),
            'Test Acc': model.score(X_test, y_test),
            'Precision': precision_score(y_test, pred),
            'Recall': recall_score(y_test, pred),
            'AUC': auc(fp, tp)
        })
    df_compare = pd.DataFrame(compare).sort_values('Test Acc', ascending=False)
    print(df_compare.to_markdown(index=False))

    plt.figure(figsize=(10, 4))
    sns.barplot(x='Model', y='Test Acc', data=df_compare)
    plt.title(f'Test Accuracy Comparison ({pheno})')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()
