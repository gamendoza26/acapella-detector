import librosa
import numpy as np

def load_and_process_audio(path, sr=22050, n_mels=128):
    """
    Load audio file, convert to mono, resample, and compute mel spectrogram.
    
    Returns:
        S_db (ndarray): Mel spectrogram in decibels
    """
    y, sr = librosa.load(path, sr=sr, mono=True)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db
