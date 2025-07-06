# ===========================================
# Heart Sound Feature Extraction Script
# Author: Mukesh
# Project: Phonocardiogram ML Prototype
# ===========================================

import os
import librosa
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from google.colab import drive

# 1. Mount Google Drive
drive.mount('/content/drive/')

# 2. Set data folder path
folder_path = '/content/drive/MyDrive/Project_B/the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data'

# 3. Collect all .wav files
wave_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
print(f"Total .wav files found: {len(wave_files)}")

# 4. Define feature extraction function
def extract_features(file_path):
    y, sr = librosa.load(file_path)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)

    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroid_mean = np.mean(spectral_centroid)

    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    bandwidth_mean = np.mean(bandwidth)

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    rolloff_mean = np.mean(rolloff)

    rmse = librosa.feature.rms(y=y)
    rmse_mean = np.mean(rmse)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1).flatten()

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(tempo)

    amp_std = np.std(y)
    skewness = skew(y)
    kurt_value = kurtosis(y)

    features = np.concatenate((
        mfccs_mean,
        [zcr_mean],
        [spectral_centroid_mean],
        [bandwidth_mean],
        [rolloff_mean],
        [rmse_mean],
        chroma_mean,
        [tempo],
        [amp_std],
        [skewness],
        [kurt_value]
    ))

    return features

# 5. Extract features for each file
feature_list = []
patient_ids = []
valves = []

for file_name in wave_files:
    file_path = os.path.join(folder_path, file_name)
    features = extract_features(file_path)
    feature_list.append(features)

    base = os.path.splitext(file_name)[0]
    parts = base.split('_')
    patient_ids.append(parts[0])
    valves.append(parts[1])

# 6. Create DataFrame
feature_df = pd.DataFrame(feature_list)

# 7. Rename columns
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

feature_df.rename(columns=rename_dict, inplace=True)

# 8. Add Patient ID and Valve columns
feature_df['Patient_ID'] = patient_ids
feature_df['Valve'] = valves

# 9. Save final CSV
output_path = '/content/drive/MyDrive/Project_B/the-circor-digiscope-phonocardiogram-dataset-1.0.3/heart_sounds_features.csv'
feature_df.to_csv(output_path, index=False)

print(f"Feature extraction complete. CSV saved to: {output_path}")