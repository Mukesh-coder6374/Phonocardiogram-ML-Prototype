"""
Phonocardiogram ML Prototype - Data Preprocessing

This script:
1. Handles missing demographic data.
2. Encodes categorical variables.
3. Reorders columns for consistency.
4. Saves a fully preprocessed dataset for training.

Author: Mukesh
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# Optional: Make NumPy output readable for debugging
np.set_printoptions(suppress=True)

# Load combined dataset
df = pd.read_csv('/storage/emulated/0/PROJECT_B/New folder/extract_df.csv')

# ---------------------------------------------------------
# STEP 1: Handle Missing Data
# ---------------------------------------------------------

# For categorical Age: use most frequent value
imputer_cat = SimpleImputer(strategy='most_frequent')
df[['Age']] = imputer_cat.fit_transform(df[['Age']])

# For numerical Height & Weight: use mean
imputer_num = SimpleImputer(strategy='mean')
df[['Height', 'Weight']] = imputer_num.fit_transform(df[['Height', 'Weight']])

# ---------------------------------------------------------
# STEP 2: Encode Categorical Variables
# ---------------------------------------------------------

# One-hot encode Age categories (drop first to avoid dummy trap)
df = pd.get_dummies(df, columns=['Age'], drop_first=True)

# Convert boolean columns to int if any
bool_cols = df.select_dtypes(include=['bool']).columns
df[bool_cols] = df[bool_cols].astype(int)

# Encode Sex: Male=1, Female=0
df['Sex'] = df['Sex'].map({'Male': 1, 'Female': 0})

# Encode Outcome: Normal=1, Abnormal=0
df['Outcome'] = df['Outcome'].map({'Normal': 1, 'Abnormal': 0})

# ---------------------------------------------------------
# STEP 3: Reorder Columns
# ---------------------------------------------------------

# Move Age dummies after Patient_ID for clarity
cols_to_move = ['Age_Child', 'Age_Infant', 'Age_Neonate']
cols = df.columns.tolist()

for col in cols_to_move:
    if col in cols:
        cols.remove(col)

insert_at = cols.index('Patient_ID') + 1
new_cols = cols[:insert_at] + cols_to_move + cols[insert_at:]
df = df[new_cols]

# ---------------------------------------------------------
# STEP 4: Save the Preprocessed Dataset
# ---------------------------------------------------------

df.to_csv("/storage/emulated/0/PROJECT_B/New folder/preprocessed_df.csv", index=False)

print("Preprocessing complete. Preprocessed file saved as: preprocessed_df.csv")