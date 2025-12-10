# A Cappella vs Non-A Cappella Audio Classifier

## What It Does

This project classifies short audio clips as a cappella or non-a cappella using mel-spectrogram features and traditional machine learning models. The system extracts frequency-based audio features, transforms them into fixed-length vectors, and applies a logistic regression classifier to predict the category. The project includes a complete training pipeline, a demo notebook for real-time inference, and evaluation tools for inspecting performance.

**Research question:** Can simple frequency-based features and a traditional ML model reliably distinguish a cappella from non-a cappella audio clips?

## Overview and Motivation

As a member of The Pitchforks, an all-male a cappella group at Duke, I often struggle to find clean voice-only recordings when preparing arrangements. Many available tracks online include hidden instrumentals or backing tracks, and manually checking every clip is time-consuming. This project explores whether mel-spectrogram frequency patterns are distinctive enough to automatically identify a cappella music. A reliable classifier could support rehearsal preparation, cataloging, and digital archiving for collegiate performance groups.

## Quick Start
```bash
git clone https://github.com/gamendoza26/acapella-detector.git
cd acapella-detector
pip install -r requirements.txt

# Run the demo notebook (upload your own audio clip when prompted)
notebooks/acapella_detector_demo.ipynb
```

## Dataset
This project uses a custom dataset of **34 audio clips** (17 a cappella, 17 non-a cappella), each 7-15 seconds long. I recorded and labeled the clips myself and even included some of my own singing from my a capella group and my piano playing.

### Preprocessing
- Convert audio to mono at 22,050 Hz
- Compute mel-spectrograms using librosa
- Convert power values to decibels for perceptual scaling
- Summarize each spectrogram by averaging across time → a 128-dimensional feature vector
  (This preserves the frequency profile while reducing model complexity)

This preprocessing pipeline is implemented in:
```bash
src/preprocessing/audio.py
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

This combination demonstrates the applied ML pipeline from raw data → features → models → evaluation in my exploration notebook.

## Evaluation & Results

Evaluation metrics directly measure the stated project objectives.

| Model | Test Accuracy | F1 Score |
|------|--------------|---------|
| Logistic Regression (L2) | 1.00 | 1.00 |
| Logistic Regression + PCA | 1.00 | 1.00 |
| Logistic Regression + Noise | 0.83 | 0.83 |
| Random Forest | 0.82 | 0.80 |

Because the dataset is small (34 total clips), I used a **70/30 train–test split** without creating a separate validation set. Allocating another partition would have reduced the number of training examples to the point where the model may not have learned meaningful structure. Instead, **hyperparameter tuning was performed on the training data**, and the final model was evaluated once on the unseen test set. This preserves learning capacity while still providing an unbiased measure of generalization performance.

### Key Insights
- Mel-spectrogram averaging provides a highly discriminative feature space, enabling perfect linear separation without deep learning.
- PCA visualization confirms that most variance aligns with a single audio dimension, explaining why logistic regression performs optimally.
- Introducing controlled noise levels reduced performance which shows the importance of recording conditions.

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
tend to cluster more closely than non-a cappella clips even in 2D. This helps explain why a simple linear classifier performs so well for this task even on limited data.

<img width="698" height="547" alt="image" src="https://github.com/user-attachments/assets/8b867961-8191-4e00-a283-5e02eccec151" />

Explained variance ratios: `[0.78457797, 0.04374359]`
The first principal component captures ~78 percent of the variance, indicating that a single dominant frequency structure differentiates a cappella vocals from instrument-backed recordings.

## Videos
**Demo Video** – https://youtu.be/DeCQaubyRe4

**Technical Walkthrough** – https://youtu.be/7qWv6Cx4wgU

## Individual Contributions
I completed this project independently. I collected and labeled the audio dataset, implemented the preprocessing pipeline, trained and evaluated the models, conducted the experiments, and wrote all documentation and code.
