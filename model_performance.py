"""
Phonocardiogram ML Prototype - Model Performance Evaluation

This script:
1. Loads the preprocessed dataset.
2. Trains Random Forest, XGBoost, and Logistic Regression.
3. Evaluates each model:
   - Confusion Matrix
   - ROC Curve
   - Feature Importance or Coefficients
4. Saves plots to /performance folder on Google Drive.

Author: Mukesh
"""

# -----------------------------------------------------------
# Imports
# -----------------------------------------------------------

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    classification_report,
    accuracy_score
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# -----------------------------------------------------------
# Load Data
# -----------------------------------------------------------

df = pd.read_csv('preprocessed_df.csv')
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=67
)

# -----------------------------------------------------------
# Create output folder
# -----------------------------------------------------------

performance_folder = '/content/drive/MyDrive/PROJECT_B/performance'
os.makedirs(performance_folder, exist_ok=True)

# -----------------------------------------------------------
# Models to Train
# -----------------------------------------------------------

models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        random_state=67
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=67
    ),
    "LogisticRegression": LogisticRegression(
        max_iter=1000,
        random_state=67
    )
}

# -----------------------------------------------------------
# Train, Evaluate, and Plot for Each Model
# -----------------------------------------------------------

for name, model in models.items():
    print(f"\n=== Training {name} ===")
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc * 100:.2f}%")
    print("Classification Report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, axs = plt.subplots(1, 3, figsize=(20, 6))

    # Confusion Matrix
    ConfusionMatrixDisplay(cm).plot(ax=axs[0])
    axs[0].set_title(f"{name} - Confusion Matrix")

    # ROC Curve
    axs[1].plot(fpr, tpr, color='darkorange', label=f"AUC = {roc_auc:.2f}")
    axs[1].plot([0, 1], [0, 1], 'k--')
    axs[1].set_xlabel('False Positive Rate')
    axs[1].set_ylabel('True Positive Rate')
    axs[1].set_title(f"{name} - ROC Curve")
    axs[1].legend(loc='lower right')

    # Feature Importances or Coefficients
    if name in ["RandomForest", "XGBoost"]:
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        axs[2].bar(range(len(importances)), importances[indices])
        axs[2].set_xticks(range(len(importances)))
        axs[2].set_xticklabels(X.columns[indices], rotation=90, fontsize=8)
        axs[2].set_title(f"{name} - Feature Importances")
    elif name == "LogisticRegression":
        coefs = model.coef_[0]
        indices = np.argsort(np.abs(coefs))[::-1]
        axs[2].bar(range(len(coefs)), coefs[indices])
        axs[2].set_xticks(range(len(coefs)))
        axs[2].set_xticklabels(X.columns[indices], rotation=90, fontsize=8)
        axs[2].set_title(f"{name} - Coefficients")

    plt.suptitle(f"{name} Performance", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    save_path = os.path.join(performance_folder, f"{name}_performance.png")
    plt.savefig(save_path)
    plt.show()

    print(f"Saved: {save_path}")