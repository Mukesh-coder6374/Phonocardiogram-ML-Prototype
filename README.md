# ❤️🔬 Phonocardiogram ML Prototype

A simple machine learning prototype for heart sound analysis using the **CIRCOR DigiScope Phonocardiogram Dataset**.
------

## 📂 Files and Scripts

| Script | Input Files | Output Files | Purpose |
|--------|--------------|---------------|---------|
| `01_extract_features.py` | `.wav` files (from Dataset) | `heart_sounds_features.csv` | Extracts MFCCs, ZCR, spectral, chroma, and other audio features |
| `02_merge_clinical_data.py` | `heart_sounds_features.csv` + `training_data.csv` | `extract_df.csv` | Merges extracted audio features with patient clinical information |
| `03_preprocess_data.py` | `extract_df.csv` | `preprocessed_df.csv` | Handles missing values, encodes categorical variables, cleans data |
| `04_train_models.py` | `preprocessed_df.csv` | Console output | Trains XGBoost, Random Forest, and Logistic Regression models |
| `05_visualize_patient_signals.py` | `.wav` files (from Dataset) | `Patient_Plots/` folder | Generates waveform, MFCC, and spectrogram plots for each patient |
| `06_evaluate_performance.py` | `preprocessed_df.csv` | `performance/` folder | Saves Confusion Matrix, ROC Curve, and Feature Importance plots |


## 📂 Dataset

**Source:** [CIRCOR DigiScope Phonocardiogram Dataset](https://physionet.org/content/circor-heart-sound/1.0.3/)

- 🎙️ **Audio:** Raw `.wav` files — heart sounds from four valves per patient.
- 📝 **Metadata:** Age, sex, height, weight, pregnancy status, outcome.

---

## ⚙️ Project Structure

```plaintext
Phonocardiogram-ML-Prototype/
│
├── preprocessing.py          # 🧹 Data cleaning & preprocessing
├── feature_extraction.py     # 🎵 Extracts MFCCs, chroma, spectral features, stats
├── model_training.py         # 🤖 ML model training & evaluation (RF, XGBoost, Logistic)
├── plots.py                  # 📊 Plots waveform, MFCC, spectrograms per patient
├── README.md                 # 📚 Project overview & instructions
├── data/
│   ├── training_data/        # 🎙️ Raw .wav audio files
│   ├── metadata.csv          # 🗂️ Original patient metadata
│   ├── extracted_features.csv # 📑 Saved extracted features
│   ├── merged_data.csv       # 🔗 Final merged & preprocessed dataset
├── performance/              # ✅ Saved model performance plots (CM, ROC, feature importance)
├── patient_plots/            # 🎨 Saved waveform, MFCC, spectrogram image
---

## ✅ How to Use

1️⃣ **Download Dataset**  
Manually download `.wav` files from the dataset link above.

2️⃣ **Organize Files**  
Place `.wav` files inside your `training_data/` folder.

3️⃣ **Run Scripts in Order**  
- `01_extract_features.py`
- `02_merge_clinical_data.py`
- `03_preprocess_data.py`
- `04_train_models.py`
- `05_visualize_patient_signals.py`
- `06_evaluate_performance.py`

4️⃣ **Check Outputs**  
- Extracted features: `heart_sounds_features.csv`
- Merged data: `extract_df.csv`
- Cleaned data: `preprocessed_df.csv`
- Model metrics printed in console
- Plots saved in `/Patient_Plots/` and `/performance/`

---

## ⚙️ Notes

- This is a **prototype** for experimenting with ML on Phonocardiogram (PCG) signals.
- Results can be improved with more advanced models and larger datasets.
- Make sure paths are correct when running locally or in Colab.

---

## ✨ Author

**Mukesh** — Biomedical Engineering & ML Enthusiast.

---
