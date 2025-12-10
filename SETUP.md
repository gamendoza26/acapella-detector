# Project Setup Instructions

## Install Dependencies

This project requires Python 3.9 or later. Install all dependencies with:

```bash
pip install -r requirements.txt
```

## Dataset Setup

This repository does **not** include raw audio files. Create the following directory structure if you want to store your own audio files to run:

```
data/raw/acapella/
data/raw/non_acapella/
```

Place your `.wav` or `.mp3` audio clips into the appropriate folders. Each clip should be relatively short (around 10 seconds long).

No processed data is included in this repository. The demo notebook preprocesses audio clips in memory at runtime and does not write anything to disk.

## Running the Classifier

Open the demo notebook and execute all cells in order:

```
notebooks/acapella_detector_demo.ipynb
```

The notebook will:

- load the pretrained logistic regression model and scaler
- accept uploaded audio clips
- preprocess them into mel-spectrogram features
- classify each clip as a cappella or non-a cappella
- return a confidence score and optionally display the spectrogram

## Google Drive / Colab Notes (If Building Your Own Model)

If running in Google Colab and your dataset is stored on Google Drive, in ```notebooks/audio_exploration.ipynb``` run:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Then update dataset paths if needed. You must provide your own clips (7-15 seconds) to reproduce the results in the exploration notebook.


