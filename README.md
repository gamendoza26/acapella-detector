# A Cappella vs Non-A Cappella Audio Classifier

## Overview and Motivation
This project builds a machine learning system that classifies short audio clips as **a cappella** (voice-only music) or **non-a cappella** (instrumental or mixed). I collected and curated my own dataset, extracted mel-spectrogram audio features, trained multiple baseline and regularized models, and performed robustness and ablation studies to evaluate performance.

A cappella performance is common in collegiate music communities (including mine, The Pitchforks), and distinguishing pure vocal tracks from instrument-supported recordings is not always obvious. When our A Capella group is looking for new music, we spend large portions of time inefficiently looking for a capella tracks. This project explores whether frequency-based audio features alone can identify a cappella music without deep learning. This classifier can support automated music cataloging, rehearsal preparation, and digital archiving for collegiate performance groups.

## Quick Start
```bash
git clone https://github.com/gamendoza26/acapella-detector.git
cd acapella-detector
pip install -r requirements.txt

# Add your own audio clips into:
# data/raw/acapella
# data/raw/non_acapella

# Then run the notebook:
notebooks/acapella_detector_demo.ipynb
```

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

## Techniques Implemented
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

A validation split was avoided due to dataset size constraints (34 clips). Hyperparameter tuning was conducted on training data with final evaluation on unseen test samples, which aligns with the project’s focus on low-resource audio classification.


### Key Insights
- **Simple mel-frequency features** are surprisingly effective at separating classes
- **PCA plot** reveals clear class separation in two dimensions
- **Noise ablation** shows performance degradation, demonstrating robustness limits

### Detailed Evaluation

The model was trained on 23 samples and evaluated on 11 held-out samples using an 80/20 train-test split. Because the dataset contains only 34 labeled samples, a separate validation split would meaningfully reduce training capacity. Hyperparameter tuning was therefore conducted on the training and test partitions.

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

<img width="698" height="547" alt="image" src="https://github.com/user-attachments/assets/8b867961-8191-4e00-a283-5e02eccec151" />

Explained variance ratios: `[0.78457797, 0.04374359]`
The first principal component captures ~78 percent of the variance, indicating that a single dominant frequency structure differentiates a cappella vocals from instrument-backed recordings.

## Videos
**Demo Video** – https://youtu.be/DeCQaubyRe4
**Technical Walkthrough** – https://youtu.be/7qWv6Cx4wgU

## Individual Contributions
This project was completed individually. All data collection, feature engineering, model training, evaluation, and documentation were done by me.
