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
##⚙️ Project Structure

Phonocardiogram-ML-Prototype/
│
├── 01_extract_features.py
├── 02_merge_clinical_data.py
├── 03_preprocess_data.py
├── 04_train_models.py
├── 05_visualize_patient_signals.py
├── 06_evaluate_performance.py
├── README.md
│
├── data/
│   ├── training_data/
│   ├── metadata.csv
│   ├── heart_sounds_features.csv
│   ├── extract_df.csv
│   ├── preprocessed_df.csv
│
├── performance/
├── patient_plots/
---

## ✅ Notes

- This is a **prototype** for experimentation only.
- Results can be improved with more data and advanced ML techniques.
- Make sure file paths match your local or cloud environment.
- Refer to the script order for running the full pipeline.
---
## 👤 Author

**Mukesh**

- 🎓 Biomedical Engineering Student  
- 💻 Exploring Machine Learning for Biomedical Signals  
- 📌 [LinkedIn](https://www.linkedin.com/in/mukesh1609)
