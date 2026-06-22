# UAV Sound Detection and Anomaly-Detection Research

A research project investigating how UAVs and other unmanned aircraft can be detected from real-world audio recordings.

The project compares multiple audio representations, supervised classifiers, anomaly-detection methods, and evaluation strategies. Its main research goal is to determine whether a system can detect UAV sounds under noisy conditions, including UAV types and recording environments that were not present during training.

A smaller part of the project explores whether a compact model can later be deployed on edge hardware.

**Project website and experimental demo:**
[roeiyanku.github.io/UAV_Sound_Classification_Project](https://roeiyanku.github.io/UAV_Sound_Classification_Project/)

**Dataset sources:**
[GitHub dataset list](DATASETS.md) · [Website dataset page](https://roeiyanku.github.io/UAV_Sound_Classification_Project/datasets/)

---

## Research Question

A standard binary classifier learns:

```text
Known UAV sounds vs. known background sounds
```

This can work well when the test recordings are similar to the training data.

The more difficult goal is:

> Can a model learn normal environmental audio and detect an unfamiliar UAV as an unusual sound?

This is why the project compares both supervised classification and anomaly-detection approaches.

---

## Project Scope

The complete pipeline includes:

* Collecting and organizing UAV audio datasets
* Processing long real-world recordings
* Manually labeling UAV activity intervals
* Splitting recordings into fixed-duration clips
* Extracting several audio representations
* Training supervised and unsupervised models
* Comparing results across datasets
* Studying data leakage and generalization
* Testing metric-learning methods
* Evaluating compact deployment models
* Presenting results through a project website

---

## Audio Processing

The preprocessing pipeline converts long recordings into model-ready examples.

```text
Raw recordings
      ↓
Manual time labels
      ↓
Mono conversion and resampling
      ↓
Fixed-duration audio clips
      ↓
Feature extraction
      ↓
Model training and evaluation
```

Processing steps include:

* Reading labeled time intervals
* Extracting UAV and background segments
* Converting multi-channel audio to mono
* Resampling recordings
* Normalizing waveform values
* Splitting long recordings into short clips
* Handling clips from several microphones
* Saving reusable datasets and model inputs

---

## Audio Representations

The project compares several ways to represent sound.

### Log-Mel Spectrograms

Transforms each waveform into a time-frequency representation and uses a CNN to learn useful patterns.

### AST Embeddings

Uses a pretrained Audio Spectrogram Transformer to extract high-level audio features.

### Wav2Vec2 Embeddings

Uses a pretrained waveform model to create representations directly from raw audio.

### YAMNet Embeddings

Uses a compact pretrained audio-event model trained on a large range of sound categories.

### Triplet-Learning Embeddings

Adapts pretrained representations so that:

* Similar clips move closer together
* Different sound classes move further apart

This is used to test whether metric learning improves UAV separation.

### Raw Waveforms

A compact neural network is also trained directly on one-second waveform inputs for the deployment experiment.

---

## Models Evaluated

### Supervised Models

Supervised models are trained using labeled UAV and background recordings.

Methods include:

* Logistic Regression
* CNN classification
* Compact 1-D neural networks

### Isolation Forest

Learns the structure of normal training examples and assigns higher anomaly scores to unusual samples.

### One-Class SVM

Creates a boundary around the normal training distribution. Samples outside the learned region are treated as anomalies.

### Teacher-Student Anomaly Detection

A student network learns to reproduce a fixed teacher network on normal audio.

During inference, a large difference between the teacher and student outputs may indicate an unusual sound.

---

## Datasets

The project uses several datasets to compare performance under different conditions.

### [Multiclass Drone Audio](https://github.com/saraalemadi/DroneAudioDataset)

Contains UAV recordings together with multiple background sound categories.

It is currently the strongest dataset in the experiments.

### [Binary Drone Audio](https://www.kaggle.com/datasets/amineipad/drone-sound-audio-detection)

Contains drone and non-drone recordings.

The current experimental run produced inconsistent results and needs to be rerun before drawing conclusions.

### [AeroSonicDB](https://www.kaggle.com/datasets/gray8ed/audio-dataset-of-low-flying-aircraft-aerosonicdb)

Contains aircraft and environmental recordings.

It is used to test whether methods that perform well on drone datasets also transfer to different aerial sounds.

### [MIMII](https://zenodo.org/records/3384388)

An industrial machine-sound dataset used as an additional anomaly-detection benchmark. The reported experiment currently uses the slider subset.

For example, a detector can learn normal machine operation and identify abnormal sounds.

### Field Recordings

The project also uses real-world recordings collected with multiple microphones and DJI UAV platforms.

These recordings require manual alignment, labeling, segmentation, and quality checking. They were collected by the project team and are not publicly distributed.

The public datasets remain subject to their original licenses and terms of use. This repository does not redistribute the complete datasets.

---

## Evaluation

The models are compared using:

* ROC-AUC
* Average Precision
* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrices
* Inference time
* Model size

For UAV detection, accuracy alone can be misleading because the datasets often contain many more background clips than UAV clips.

Recall, precision, F1-score, ROC-AUC, and Average Precision are therefore especially important.

---

## Preliminary Results

Best preliminary result reported for each dataset:

| Dataset                | Best method                               | ROC-AUC | Anomaly F1 |
| ---------------------- | ----------------------------------------- | ------: | ---------: |
| Multiclass Drone Audio | AST triplet embeddings + Isolation Forest |   1.000 |      0.806 |
| MIMII Slider           | AST triplet embeddings + Isolation Forest |   1.000 |      0.947 |
| AeroSonicDB            | Wav2Vec2 embeddings + Isolation Forest    |   0.724 |      0.365 |
| Binary Drone Audio     | AST embeddings + Isolation Forest         |   0.468 |      0.016 |

These results are preliminary.

The large differences between datasets show that strong performance on one dataset does not guarantee reliable detection in another recording environment.

---

## Evaluation Limitations

Most current experiments use a random clip-level split.

This can cause clips from the same recording session to appear in both training and testing.

For example:

```text
Original recording
├── clip 1 → training
├── clip 2 → training
└── clip 3 → testing
```

Because nearby clips may share the same microphone, environment, UAV, and background noise, the test results may be overly optimistic.

### Planned Evaluation Improvements

#### Recording-Session Split

All clips from one recording session stay in the same dataset split.

#### Leave-One-UAV-Out

One complete UAV model or recording session is excluded from training and used only for testing.

#### Unseen Negative Categories

New non-UAV sounds such as alarms, engines, tools, speech, and aircraft are reserved for final testing.

These experiments are necessary before claiming that the system can detect an unseen UAV.

---

## TinyML Deployment Experiment

TinyML is a smaller deployment branch of the overall research project.

A compact **mini-SE-Net** was trained directly on one-second, 16 kHz waveforms and exported to full-int8 TensorFlow Lite.

Reported measurements:

| Measurement             |   Value |
| ----------------------- | ------: |
| Parameters              |   7,350 |
| Model size              | 22.6 KB |
| Keras test accuracy     |  97.84% |
| ROC-AUC                 |  0.9950 |
| Drone recall            |   0.870 |
| Drone F1                |   0.821 |
| Notebook TFLite latency | 1.32 ms |

The model has not yet been fully benchmarked on the final physical edge device.

Prediction consistency between the original Keras model, Python TensorFlow Lite inference, and browser inference is still being validated.

---

## Browser Demo

The project website contains an experimental local audio demo.

The demo:

* Accepts uploaded recordings
* Supports microphone recording
* Converts audio to mono
* Resamples it to 16 kHz
* Splits it into one-second windows
* Runs a TensorFlow Lite model in the browser
* Displays scores for each audio window

The audio is processed locally and is not uploaded to a server.

The browser demo is a deployment demonstration and not the main research result.

---

## Running the Project

Clone the repository:

```bash
git clone https://github.com/roeiyanku/UAV_Sound_Classification_Project.git
cd UAV_Sound_Classification_Project
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Main libraries include:

* NumPy
* Pandas
* Librosa
* Scikit-learn
* TensorFlow
* PyTorch
* Hugging Face Transformers
* Matplotlib
* Joblib

Most experiments are designed to run in Google Colab.

Update the project and dataset paths before running the notebooks:

```python
PROJECT_DIR = "/content/drive/MyDrive/Final Project RMOT"
```

Run the notebooks in the order required for:

1. Data preparation
2. Dataset splitting
3. Feature extraction
4. Model training
5. Anomaly detection
6. Model comparison
7. Deployment experiments

---

## Website

The GitHub Pages website files are stored inside:

```text
docs/
```

Run the website locally with:

```bash
cd docs
python -m http.server 8000
```

Then open:

```text
http://localhost:8000
```

---

## Main Limitations

An unusual sound is not automatically a UAV.

An anomaly detector may also react to:

* Alarms
* Cars and engines
* Aircraft
* Construction equipment
* Speech
* Wind
* Recording artifacts
* Unfamiliar environmental sounds

A reliable real-world detector requires diverse training data, difficult negative examples, unseen-UAV evaluation, and testing on the final recording hardware.

---

## Next Steps

* Rerun inconsistent dataset experiments
* Use recording-session splits
* Perform leave-one-UAV-out testing
* Add more difficult background categories
* Compare different UAV models and distances
* Test performance under different signal-to-noise ratios
* Add overlapping audio windows
* Add temporal smoothing
* Improve probability calibration and thresholds
* Validate the browser inference pipeline
* Benchmark the compact model on physical edge hardware
* Test whether anomaly detection can distinguish UAVs from other unusual sounds

---

## Team

### Roei Yanku — ML and Audio Engineering

Audio processing, feature extraction, model training, experiment integration, TinyML export, and browser-demo implementation.

### Melissa Liebowitz — Research and Experimentation

Field data collection, experiment execution, evaluation support, research, and project documentation.

### Tal Kfir — Team Lead

Project coordination, planning, task assignment, and milestone tracking.

### Or Anidjar — Project Oversight

Overall responsibility, direction, and project-level oversight.

### Boaz Ben-Moshe — Laboratory and Field Support

Provided access to research equipment, laboratory facilities, DJI platforms, and field-recording support.

---

## Project Status

This is an active research project.

The main focus is comparing audio representations and detection methods while building a more reliable evaluation protocol for unseen UAVs and real-world recording conditions.

The current results should be treated as preliminary until recording-level and leave-one-UAV-out experiments are completed.
