"""
Phonocardiogram ML Prototype - Feature Renaming & Dataset Merge

This script:
1. Renames extracted feature columns for clarity.
2. Combines all valve recordings per patient by averaging.
3. Merges audio features with patient demographics & labels.
4. Exports the cleaned dataframe for model training.

Author: Mukesh
"""

import pandas as pd
import numpy as np

# Load extracted features
df = pd.read_csv('/storage/emulated/0/PROJECT_B/heart_sounds_features (1).csv')

# Rename columns: MFCCs, chroma, and other audio features
rename_dict = {f"{i}": f"mfcc_{i+1}" for i in range(13)}
rename_dict.update({
    "13": "zcr",
    "14": "spectral_centroid",
    "15": "bandwidth",
    "16": "rolloff",
    "17": "rmse",
})
for i in range(12):
    rename_dict[str(18 + i)] = f"chroma_{i+1}"
rename_dict.update({
    "30": "tempo",
    "31": "amp_std",
    "32": "skewness",
    "33": "kurt"
})

df_renamed = df.rename(columns=rename_dict)

# Reorder: Patient_ID, Valve first
cols = list(df_renamed.columns)
other_cols = [c for c in cols if c not in ['Patient_ID', 'Valve']]
new_cols_order = ['Patient_ID', 'Valve'] + other_cols
df_renamed = df_renamed[new_cols_order]

# Sort and reset index
df_renamed.sort_values(by='Patient_ID', inplace=True)
df_renamed.reset_index(drop=True, inplace=True)

# Save the cleaned feature file
df_renamed.to_csv('/storage/emulated/0/PROJECT_B/New folder/Heart_sound_features.csv', index=False)

# Group by Patient_ID → average features across valves
df = pd.read_csv('/storage/emulated/0/PROJECT_B/New folder/Heart_sound_features.csv')
feature_cols = (
    [f"mfcc_{i}" for i in range(1, 14)] +
    [f"chroma_{i}" for i in range(1, 13)] +
    ["zcr", "spectral_centroid", "bandwidth", "rolloff", "rmse",
     "tempo", "amp_std", "skewness", "kurt"]
)

combined = df.groupby("Patient_ID")[feature_cols].mean().reset_index()
combined = combined.rename(columns={col: f"{col}_combined" for col in feature_cols})

# Load patient metadata (demographics, label)
df1 = pd.read_csv('/storage/C7A5-1A19/Music/training_data (1).csv')
selected_columns = [
    "Patient ID",
    "Age",
    "Sex",
    "Height",
    "Weight",
    "Pregnancy status",
    "Outcome"
]
df2 = df1[selected_columns]

# Drop rows with pregnancy TRUE
df2 = df2[df2["Pregnancy status"] != True]
df2 = df2.drop(columns=["Pregnancy status"])
df2 = df2.rename(columns={"Patient ID": "Patient_ID"}).reset_index(drop=True)

# Merge metadata with audio features
combined = combined[combined['Patient_ID'].isin(df2['Patient_ID'])].reset_index(drop=True)
Dataframe = pd.merge(df2, combined, on='Patient_ID')

# Move Outcome to last column
outcome_col = Dataframe.pop('Outcome')
Dataframe['Outcome'] = outcome_col

print(Dataframe.head())

# Save the final dataset
Dataframe.to_csv("/storage/emulated/0/PROJECT_B/New folder/extract_df.csv", index=False)