# Project Setup Instructions

## Install Dependencies

This project requires Python 3.9 or later. Install all dependencies with:

```bash
pip install -r requirements.txt
```

## Dataset Setup

This repository does **not** include raw audio files. Create the following directory structure:

```
data/raw/acapella/
data/raw/non_acapella/
```

Place your `.wav` or `.mp3` audio clips into the appropriate folders. Each clip should be 7–15 seconds long.

Processed mel-spectrogram features will be generated automatically the first time the notebook is run and stored in:

```
data/processed/
```

No manual preprocessing is required.

## Running the Classifier

Open the main notebook and execute all cells in order:

```
notebooks/acapella_detector_demo.ipynb
```

The notebook will:

- load raw audio files
- compute mel-spectrogram features using `librosa`
- train and evaluate logistic regression and random forest models
- perform hyperparameter tuning
- generate confusion matrices and PCA visualizations

## Google Drive / Colab Notes (If Applicable)

If running in Google Colab and your dataset is stored on Google Drive, run:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Then update dataset paths if needed. Raw audio data is not included due to copyright restrictions. You must provide your own clips to reproduce the results.
