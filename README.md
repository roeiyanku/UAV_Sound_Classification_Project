# UAV Sound Detection and Anomaly-Detection Research

Final RMOT research project on acoustic UAV detection across six datasets, using pretrained audio embeddings, supervised and unsupervised classifiers, triplet fine-tuning, cross-dataset transfer, and a compact TinyML deployment model.

**Project website:**
[roeiyanku.github.io/UAV_Sound_Classification_Project](https://roeiyanku.github.io/UAV_Sound_Classification_Project/)

**Datasets page:**
[roeiyanku.github.io/UAV_Sound_Classification_Project/datasets/](https://roeiyanku.github.io/UAV_Sound_Classification_Project/datasets/)

**Final paper:**
[Acoustic UAV Detection with Pretrained Audio Embeddings](https://roeiyanku.github.io/UAV_Sound_Classification_Project/paper/UAV_Classification_paper.pdf/)

## Final benchmark

The study evaluates six corpora: Binary Drone Audio, AeroSonicDB, Multiclass Drone Audio, Electric UAVs, FPV, and UAV. The first three are public datasets; the final three were recorded by the team.

Best configuration by AUC:

| Dataset | Best configuration | AUC |
| --- | --- | ---: |
| Binary Drone Audio | AST · Logistic Regression | 0.999 |
| AeroSonicDB | YAMNet · Logistic Regression | 0.954 |
| Multiclass Drone Audio | AST · Logistic Regression | 1.000 |
| Electric UAVs | AST-triplet · Logistic Regression | 0.913 |
| FPV | AST-triplet · Logistic Regression | 0.922 |
| UAV | YAMNet-triplet · Logistic Regression | 0.759 |
| Train-all → FPV | AST-triplet · Logistic Regression | 1.000 |

The strongest unsupervised detectors remain close to supervised performance on the three field datasets. Triplet fine-tuning helps most consistently on field recordings and cross-dataset transfer, while its effect is mixed on public corpora.

## Edge deployment

A raw-waveform mini-SE-Net was quantized to full int8 TensorFlow Lite:

- ~7.4k parameters
- ~22.6 KB model size
- AUC 0.68–0.99 across the single datasets
- ~1.5–2.5 ms per one-second sample on the Colab host CPU

The physical target-device benchmark remains future work.

## Website

The GitHub Pages site keeps the original project structure, including:

- Overview and motivation
- Method
- Field-recording media
- Dedicated datasets page
- Browser TinyML demo
- Final benchmark tables
- Supervised vs. unsupervised comparison
- Triplet fine-tuning findings
- TinyML results
- Final paper page
- Team section

GitHub Pages files are stored under `docs/`.

## Team

- Roei Yanku — ML and Audio Engineering
- Melissa Liebowitz — Research and Experimentation
- Tal Kfir — Team Lead
- Or Anidjar — Project Oversight
- Boaz Ben-Moshe — Laboratory and Field Support
