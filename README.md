<img width="487" height="404" alt="image" src="https://github.com/user-attachments/assets/463f2250-cfa6-444f-9f53-5fd79ffddfe9" /># A Cappella vs Non-A Cappella Audio Classifier 🎵

## What It Does
This project builds a machine learning system that can automatically classify short audio clips as **a cappella** (voice-only music) or **non-a cappella** (instrumental or mixed). I collected and curated my own dataset, extracted mel-spectrogram audio features, trained multiple baseline and regularized models, and performed robustness and ablation studies to evaluate performance.

## Quick Start
```bash
git clone https://github.com/gamendoza26/acapella-detector.git
cd acapella-detector
pip install -r requirements.txt

# Add your own audio clips into:
# data/raw/acapella
# data/raw/non_acapella

# Then run the notebook:
notebooks/acapella_detector_experiments.ipynb
```

## Videos
**Demo Video** – *coming soon*  
**Technical Walkthrough** – *coming soon*

## Dataset
This project uses a custom dataset of **34 audio clips** (17 a cappella, 17 non-a cappella), each 7-15 seconds long. I recorded and labeled the clips myself and cannot include the raw files in this repository due to copyright restrictions.

### Preprocessing
- Convert audio to mono at 22,050 Hz
- Compute **mel-spectrograms** using `librosa`
- Summarize each spectrogram by averaging across time → **128-dimensional feature vector**
- This enables lightweight training while preserving core frequency patterns

This preprocessing pipeline is implemented in:
```bash
src/preprocessing/audio.py
src/preprocessing/dataset.py
```

## Models and Techniques Implemented

### Core ML Workflow
| Component | Status |
|----------|-------|
| Baseline classifier (Logistic Regression) | ✅ |
| Regularization (L1, L2, Early Stopping) | ✅ |
| Random Forest comparison | ✅ |
| Hyperparameter tuning (C sweep) | ✅ |
| PCA Dimensionality Reduction | ✅ |
| Noise robustness experiment | ✅ |
| Multiple performance metrics (Accuracy, Precision, Recall, F1) | ✅ |
| Confusion matrices + error analysis | ✅ |

This combination demonstrates a complete applied ML pipeline from raw data → features → models → evaluation.

## Evaluation & Results

| Model | Test Accuracy | F1 Score |
|------|--------------|---------|
| Logistic Regression (L2) | 1.00 | 1.00 |
| Logistic Regression + PCA | 1.00 | 1.00 |
| Logistic Regression + Noise | 0.83 | 0.83 |
| Random Forest | 0.82 | 0.80 |

### Key Insights
- **Simple mel-frequency features** are surprisingly effective at separating classes
- **PCA plot** reveals clear class separation in two dimensions
- **Noise ablation** shows performance degradation, demonstrating robustness limits

### Detailed Evaluation

The model was trained on 23 samples and evaluated on 11 held-out samples using an 80/20 train-test split.

#### Clean Test Performance (Logistic Regression, L2 Regularization)

**Metrics**
- Accuracy: 1.00  
- Precision (macro): 1.00  
- Recall (macro): 1.00  
- F1 Score (macro): 1.00  

**Confusion Matrix**

```text
                 Predicted Non-A Cappella   Predicted A Cappella
True Non-A Cappella            6                     0
True A Cappella                0                     5
```

**PCA Projection of Audio Features**
The mel-frequency features exhibit clear separability between the two classes.  
This 2D projection of the 128-dimensional feature vectors shows that a cappella clips
tend to cluster separately from non-a cappella clips, which explains why a simple
linear classifier performs so well.

<p align="center">
  <img src="docs/pca_projection.png" alt="PCA projection of audio features" width="500">
</p>

Explained variance ratios: `[0.78457797, 0.04374359]`
The first principal component captures ~78 percent of the variance, indicating that a single dominant frequency structure differentiates a cappella vocals from instrument-backed recordings.


## Rubric Items Claimed (Machine Learning)

- Train/test split with documented ratios
- Baseline vs advanced model comparison
- Regularization techniques (L1, L2, early stopping)
- Hyperparameter tuning
- Feature engineering via mel-spectrogram aggregation
- Dimensionality reduction (PCA)
- Noise-based robustness / ablation study
- Multiple evaluation metrics
- Error analysis
- Original dataset collection
- Solo project credit

## Project Motivation
A cappella performance is common in collegiate music communities (including mine, The Pitchforks), and distinguishing pure vocal tracks from instrument-supported recordings is not always obvious. This project explores whether frequency-based audio features alone can identify a cappella music without deep learning. This classifier can support automated music cataloging, rehearsal preparation, and digital archiving for collegiate performance groups.

## Individual Contributions
This project was completed individually. All data collection, feature engineering, model training, evaluation, and documentation were done by me.
