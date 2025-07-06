"""
Phonocardiogram ML Prototype - Model Training & Evaluation

This script:
1. Loads the preprocessed dataset.
2. Scales the features.
3. Trains:
   - XGBoost Classifier
   - Random Forest Classifier
   - Logistic Regression
4. Evaluates each model with confusion matrix, accuracy, and CV score.
5. Plots XGBoost feature importance.

Author: Mukesh
"""

# -----------------------------------------------------------
# Imports
# -----------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, plot_importance
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# Load Data
# -----------------------------------------------------------

df = pd.read_csv('preprocessed_df.csv')

# Split features & target
X = df.iloc[:, 1:-1].values  # skip Patient_ID
y = df.iloc[:, -1].values

# -----------------------------------------------------------
# Feature Scaling
# -----------------------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=67
)

# -----------------------------------------------------------
# XGBoost Classifier
# -----------------------------------------------------------

print("\n=== XGBoost Classifier ===\n")

xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=67
)

xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))
print("\nAccuracy: {:.2f}%".format(accuracy_score(y_test, y_pred_xgb) * 100))
print("\nClassification Report:\n", classification_report(y_test, y_pred_xgb))

cv_scores_xgb = cross_val_score(xgb, X_scaled, y, cv=5)
print("\nMean CV Accuracy: {:.2f}%".format(np.mean(cv_scores_xgb) * 100))

# Plot XGBoost Feature Importance
plt.figure(figsize=(12, 8))
plot_importance(xgb)
plt.title('Feature Importance - XGBoost')
plt.show()

# -----------------------------------------------------------
# Random Forest Classifier
# -----------------------------------------------------------

print("\n=== Random Forest Classifier ===\n")

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=True,
    random_state=67
)

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("\nAccuracy: {:.2f}%".format(accuracy_score(y_test, y_pred_rf) * 100))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))

# -----------------------------------------------------------
# Logistic Regression
# -----------------------------------------------------------

print("\n=== Logistic Regression ===\n")

lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
print("\nAccuracy: {:.2f}%".format(accuracy_score(y_test, y_pred_lr) * 100))
print("\nClassification Report:\n", classification_report(y_test, y_pred_lr))