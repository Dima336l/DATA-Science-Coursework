"""Standalone script to export report figures. Run after loading data and fitting models.
   Requires: df (or load csgo_games.csv), df_features, y_test, y_pred_lr, and preprocessing.
   Best used by running the equivalent code in the notebook's 'Export Report Figures' cell.
"""
import os
os.makedirs("figures", exist_ok=True)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Load and minimal preprocessing
df = pd.read_csv("csgo_games.csv")
df = df.dropna(subset=["winner"])
target = df["winner"]
df = df.select_dtypes(include=["number"]).drop(columns=["t1_points", "t2_points"], errors="ignore")
imp = SimpleImputer(strategy="median")
X = pd.DataFrame(imp.fit_transform(df), columns=df.columns)
le = LabelEncoder()
y = le.fit_transform(target)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=123, stratify=y)
lr = LogisticRegression(max_iter=1000, random_state=123)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# 1. Class distribution
plt.figure(figsize=(8, 5))
counts = target.value_counts()
plt.bar(counts.index, counts.values)
plt.title("Distribution of Match Winners")
plt.xlabel("Winner")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("figures/fig_class_distribution.png", dpi=150)
plt.close()

# 2. Correlation heatmap
corr_cols = [c for c in ["t1_world_rank", "t2_world_rank", "t1_h2h_win_perc", "t2_h2h_win_perc",
                         "t1_player1_rating", "t2_player1_rating"] if c in X.columns]
if len(corr_cols) >= 2:
    plt.figure(figsize=(10, 8))
    sns.heatmap(X[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap of Key Features")
    plt.tight_layout()
    plt.savefig("figures/fig_correlation_heatmap.png", dpi=150)
    plt.close()

# 3. Confusion matrix
cm = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Logistic Regression")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig("figures/fig_confusion_matrix_lr.png", dpi=150)
plt.close()

print("Figures saved to figures/")
