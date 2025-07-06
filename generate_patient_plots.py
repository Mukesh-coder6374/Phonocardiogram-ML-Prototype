"""
Phonocardiogram ML Prototype - Generate Patient Audio Visualizations

This script mounts Google Drive, loads all patient .wav files,
and generates plots for:
 - Time-domain waveform
 - MFCCs
 - Spectrogram

Each patient’s plots for four valves (AV, MV, PV, TV) are saved
as combined images in Google Drive.

Author: Mukesh
"""

import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Paths
folder_path = '/content/drive/MyDrive/Project_B/the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data'
save_path = '/content/drive/MyDrive/Project_B/Patient_Plots_All'
os.makedirs(save_path, exist_ok=True)

# List all .wav files in dataset
wave_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]

# Get unique patient IDs based on filename convention: PatientID_Valve.wav
patient_ids = set([f.split('_')[0] for f in wave_files])

# Loop over all unique patients
for patient_id in sorted(patient_ids):
    plt.figure(figsize=(15, 20))  # 4 rows x 3 columns for AV, MV, PV, TV

    for i, valve in enumerate(['AV', 'MV', 'PV', 'TV']):
        # Find the file matching patient and valve
        matching_files = [f for f in wave_files if f.startswith(f"{patient_id}_{valve}")]
        if not matching_files:
            continue  # Skip missing valve recordings

        file_path = os.path.join(folder_path, matching_files[0])
        y, sr = librosa.load(file_path)

        # 1) Waveform
        plt.subplot(4, 3, i * 3 + 1)
        librosa.display.waveshow(y, sr=sr)
        plt.title(f'Patient {patient_id} - {valve} - Waveform')

        # 2) MFCCs
        plt.subplot(4, 3, i * 3 + 2)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        librosa.display.specshow(mfccs, x_axis='time', sr=sr)
        plt.colorbar()
        plt.title(f'Patient {patient_id} - {valve} - MFCC')

        # 3) Spectrogram
        plt.subplot(4, 3, i * 3 + 3)
        D = np.abs(librosa.stft(y))
        DB = librosa.amplitude_to_db(D, ref=np.max)
        librosa.display.specshow(DB, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Patient {patient_id} - {valve} - Spectrogram')

    plt.tight_layout()
    output_file = os.path.join(save_path, f"{patient_id}_plots.png")
    plt.savefig(output_file)
    plt.close()
    print(f"Saved plots for Patient {patient_id} at: {output_file}")