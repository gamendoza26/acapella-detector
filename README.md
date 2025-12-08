# A Cappella vs Non-A Cappella Audio Classifier 🎵

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
