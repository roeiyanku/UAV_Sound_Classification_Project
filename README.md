# UAV Sound Classification Pipeline

An audio machine-learning project for detecting and classifying **UAV/drone sounds in real-world recordings**.

The project compares several supervised and anomaly-detection approaches:

* CNN on log-mel spectrograms
* AST embeddings with Logistic Regression
* Wav2Vec2 embeddings with Logistic Regression
* One-Class SVM
* Isolation Forest
* Teacher-Student anomaly detection

The pipeline was organized with an emphasis on clean and maintainable code:

* Clear notebook sections
* Shared configuration values
* Reusable helper functions
* Meaningful variable names
* Separation between preprocessing, training, and evaluation

---

## Project Goals

The main goal is to compare different machine-learning approaches for detecting UAV sounds in noisy, real-world audio recordings.

The project focuses on:

* Processing and segmenting raw audio recordings
* Extracting meaningful audio features
* Distinguishing UAV sounds from background noise
* Comparing supervised and anomaly-detection models
* Evaluating all models with consistent metrics
* Building a readable and reusable training pipeline

---

## Models Included

### 1. CNN on Log-Mel Spectrograms

Converts each audio clip into a log-mel spectrogram and trains a Convolutional Neural Network to classify UAV and non-UAV sounds.

### 2. AST with Logistic Regression

Uses a pretrained Audio Spectrogram Transformer to extract audio embeddings. A Logistic Regression classifier is then trained on the extracted features.

### 3. Wav2Vec2 with Logistic Regression

Uses a pretrained Wav2Vec2 model to extract embeddings directly from raw audio waveforms, followed by Logistic Regression classification.

### 4. One-Class SVM

Trained mainly on background or non-UAV recordings. Audio clips that differ from the learned background distribution are identified as possible UAV sounds.

### 5. Isolation Forest

An anomaly-detection model that learns the characteristics of background audio and detects unusual sounds as potential UAV activity.

### 6. Teacher-Student Anomaly Detection

A student model learns to reproduce the output of a fixed teacher model on normal background audio. A large difference between their outputs is used as an anomaly score.

---

## Evaluation

The models are compared using:

* ROC-AUC
* Accuracy
* Precision
* F1-score
* Sensitivity / Recall
* Confusion matrix
* Training and inference time
* Estimated saved model size

---

## Project Structure

```text
uav-sound-classification/
├── notebooks/
│   ├── data_preparation.ipynb
│   ├── training_pipeline.ipynb
│   └── model_comparison.ipynb
├── src/
│   ├── config.py
│   ├── audio_processing.py
│   ├── features.py
│   ├── models.py
│   ├── anomaly.py
│   └── evaluate.py
├── data/
│   ├── uav/
│   └── background/
├── models/
├── requirements.txt
├── .gitignore
└── README.md
```

Most of the current implementation is contained in the notebooks. Reusable functionality can later be moved into the `src/` directory.

---

## Dataset Layout

The pipeline expects audio files to be organized into two main classes:

```text
data/
├── uav/
│   └── ... .wav files
└── background/
    └── ... .wav files
```

* `uav/` contains audio clips where a UAV can be heard.
* `background/` contains environmental sounds without a UAV.

Example path in Google Colab:

```python
DATA_DIR = "/content/drive/MyDrive/uav_sound_project/data"
```

Update this path to match the location of your dataset.

---

## Audio Preparation

Long recordings are divided into shorter audio clips before training.

Each clip is assigned one of two labels:

* `1` — UAV sound
* `0` — Background or non-UAV sound

The preparation pipeline may include:

* Reading labeled time intervals
* Splitting recordings into fixed-length clips
* Resampling audio
* Converting stereo audio to mono
* Normalizing waveform values
* Saving the clips into class folders

---

## Installation

Create a Python environment and install the required dependencies:

```bash
pip install -r requirements.txt
```

The main libraries include:

* NumPy
* Pandas
* Librosa
* Scikit-learn
* TensorFlow or PyTorch
* Hugging Face Transformers
* Matplotlib

---

## How to Run

### Option 1: Google Colab

1. Upload the notebooks to Google Drive.
2. Open the main notebook with Google Colab.
3. Mount Google Drive.
4. Update the `DATA_DIR` variable.
5. Run the notebook from top to bottom.

### Option 2: Local Machine

1. Clone the repository:

```bash
git clone <repository-url>
cd uav-sound-classification
```

2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Place the dataset inside the expected folder structure.
4. Update the dataset path.
5. Run the notebooks using Jupyter.

---

## Clean Code Principles

The project follows several basic clean-code practices:

* Avoid repeating configuration values
* Keep functions short and focused
* Use descriptive names
* Separate preprocessing, feature extraction, training, and evaluation
* Use the same evaluation process for every model
* Make it possible to replace one model without changing the entire pipeline

For example, shared values are defined once:

```python
SAMPLE_RATE = 16000
CLIP_SECONDS = 5
BATCH_SIZE = 32
RANDOM_STATE = 42
```

This makes experiments easier to reproduce and update.

---

## Future Improvements

Possible improvements include:

* Collecting more UAV recordings from different environments
* Testing different UAV models and flight distances
* Adding audio augmentation such as noise, pitch shifting, and time shifting
* Evaluating performance at different signal-to-noise ratios
* Moving reusable notebook code into the `src/` directory
* Adding command-line training and inference scripts
* Adding real-time microphone detection
* Testing deployment on edge devices
* Adding unit tests
* Saving experiment results automatically
* Creating a simple web interface for uploading and analyzing audio

---

## Notes

The project uses pretrained Hugging Face models such as AST and Wav2Vec2. During the first run, their model weights will be downloaded automatically.

These models can require significant memory and may run faster with GPU acceleration.

Performance may also vary depending on:

* UAV distance
* UAV model
* Wind and environmental noise
* Recording device
* Clip duration
* Signal-to-noise ratio

---

## Author

**Roei Yanku**

Computer Science student interested in machine learning, audio processing, UAV systems, and practical AI applications.
