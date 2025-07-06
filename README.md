# ❤️🔬 Phonocardiogram ML Prototype

A simple machine learning prototype for heart sound analysis using the **CIRCOR DigiScope Phonocardiogram Dataset**.

---

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
