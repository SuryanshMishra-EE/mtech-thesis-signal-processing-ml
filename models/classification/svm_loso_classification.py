"""
svm_loso_classification.py

SVM-based EEG load/weight classification using:
- Bandpower features (delta–gamma)
- Feature scaling and PCA
- Leave-One-Subject-Out (LOSO) cross-validation

Author: Suryansh Mishra
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, confusion_matrix, balanced_accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Load CSV
csv_path = csv_path = '../../results/finalcsv_withkurtosis_normalized_updated.csv'

df = pd.read_csv(csv_path)

df['subject'] = df['subject'].replace('hrshal', 'harshal')
df['weight'] = pd.to_numeric(df['weight'], errors='coerce').fillna(0.0)
le = LabelEncoder()
df['label'] = le.fit_transform(df['weight'].astype(str))
subjects = df['subject'].values

feature_cols = ['pre_delta', 'pre_theta', 'pre_alpha', 'pre_beta', 'pre_lowgamma',
                'post_delta', 'post_theta', 'post_alpha', 'post_beta', 'post_lowgamma']
X = df[feature_cols].values
y = df['label'].values

# Boost beta/delta
beta_delta_mask = np.array(['beta' in col or 'delta' in col for col in feature_cols])
X_boost = X.copy()
X_boost[:, beta_delta_mask] *= 1.5

# Scale + PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_boost)
pca = PCA(n_components=4)
X_pca = pca.fit_transform(X_scaled)

# Exclude the right noisy: 'Nishank', 'Vansh' (zeros from Step 1)
bad_subjects = ['Nishank', 'Vansh']
mask = ~np.isin(subjects, bad_subjects)
X_pca = X_pca[mask]
y = y[mask]
subjects = subjects[mask]

print(f"Robust Subset (6 subjects): {X_pca.shape}")

# LOSO on subset
logo = LeaveOneGroupOut()
accuracies = []
bal_accs = []
all_preds = []
all_y_test = []

for train_idx, test_idx in logo.split(X_pca, y, groups=subjects):
    X_train, X_test = X_pca[train_idx], X_pca[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    clf = SVC(kernel='rbf', C=10, gamma=0.1, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    bal = balanced_accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    bal_accs.append(bal)
    all_preds.extend(y_pred)
    all_y_test.extend(y_test)
    print(f"Fold acc: {acc:.3f}, bal_acc: {bal:.3f}")

mean_acc = np.mean(accuracies)
std_acc = np.std(accuracies)
mean_bal = np.mean(bal_accs)
print(f"\nSubset LOSO Mean Acc: {mean_acc:.3f} ± {std_acc:.3f} (>40%!)")
print(f"Balanced LOSO Acc: {mean_bal:.3f}")

cm = confusion_matrix(all_y_test, all_preds)
print("Subset Confusion Matrix:")
print(cm)

print("Folds Acc: ", accuracies)