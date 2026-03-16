# Pump Sound Classification Pipeline

A clean and practical audio-ML project for **normal vs. abnormal pump sound classification**.

This project compares several approaches on pump audio data:
- CNN on log-mel spectrograms
- AST embeddings + Logistic Regression
- Wav2Vec2 embeddings + Logistic Regression
- One-Class SVM
- Isolation Forest
- Teacher-Student anomaly detection

The notebook was organized with an emphasis on **clean code principles**:
- clear sectioning
- shared constants instead of magic numbers
- reusable helper functions
- meaningful variable names
- one main responsibility per block

---

## Project goals

The main goal is to compare different supervised and anomaly-detection approaches for audio classification and see which method works best on abnormal pump sounds.

The project focuses on:
- feature extraction from raw audio
- training multiple model types
- evaluating them with the same metrics
- keeping the notebook readable and easy to maintain

---

## Models included

### 1. CNN on log-mel spectrograms
Turns each waveform into a spectrogram image and trains a CNN.

### 2. AST + Logistic Regression
Uses a pretrained Audio Spectrogram Transformer to extract embeddings, then trains a simple classifier.

### 3. Wav2Vec2 + Logistic Regression
Uses pretrained Wav2Vec2 embeddings from raw waveform audio.

### 4. One-Class SVM
Trained only on **normal** samples, then tries to detect abnormal ones as anomalies.

### 5. Isolation Forest
Another anomaly-detection model trained on normal samples.

### 6. Teacher-Student anomaly detection
Learns normal behavior by comparing teacher and student outputs and uses reconstruction / imitation error as an anomaly signal.

---

## Evaluation

The notebook compares models with:
- ROC-AUC
- Accuracy
- F1-score
- Sensitivity / Recall
- Training or inference time
- Estimated saved model size

---

## Project structure

```text
pump-sound-project/
├── notebooks/
│   ├── data_download.ipynb
│   ├── training_pipeline.ipynb
│   └── training_pipeline_clean_code_ready.ipynb
├── src/
│   ├── config.py
│   ├── features.py
│   ├── models.py
│   ├── anomaly.py
│   └── evaluate.py
├── requirements.txt
├── .gitignore
└── README.md
```

Right now, the main work is in the notebook. A strong next step is moving repeated logic into `src/` files.

---

## Dataset layout

The notebook expects a structure like this:

```text
data/
├── normal/
│   └── ... .wav files
└── abnormal/
    └── ... .wav files
```

Example path in Colab:

```python
DATA_DIR = "/content/drive/MyDrive/pump_sound_project/data"
```

Change that path to match your own Drive folder.

---

## Installation

Create an environment and install dependencies:

```bash
pip install -r requirements.txt
```

---

## How to run

### Option 1: Google Colab
1. Upload the notebook to Google Drive.
2. Open it with Google Colab.
3. Mount Google Drive if your data is stored there.
4. Update `DATA_DIR`.
5. Run the notebook from top to bottom.

### Option 2: Local machine / Jupyter
1. Clone the repository.
2. Install requirements.
3. Make sure your dataset follows the expected folder structure.
4. Update `DATA_DIR` in the notebook.
5. Run the notebook.

---

## Clean code notes

This project tries to follow a few simple clean-code ideas:
- avoid repeating values like sample rate and clip length
- keep helper functions short and focused
- use descriptive names like `file_paths`, `labels`, `cnn_model`
- separate setup, features, training, and evaluation
- make it easy to replace one model without changing the whole notebook

Example:
Instead of repeating values like `16000` or `5` in many places, define constants such as:

```python
SAMPLE_RATE = 16000
CLIP_SECONDS = 5
BATCH_SIZE = 32
```

---

## Future improvements

Good next steps for making the project look even stronger on GitHub:
- move reusable code into `src/`
- save trained models into a `models/` folder
- add sample results / screenshots
- add command-line training scripts
- add a small `inference.ipynb` notebook
- add unit tests for helper functions

---

## Notes

Some parts of the project use pretrained Hugging Face audio models such as AST and Wav2Vec2, so the first run may take time because model weights need to download.

---

## Author

Roei Yanku

If you want, later you can also add your GitHub and LinkedIn links here.
