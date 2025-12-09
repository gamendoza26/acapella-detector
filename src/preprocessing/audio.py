import os
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np

LABEL_MAP = {"non_acapella": 0, "acapella": 1}

def extract_mel_features(
    path: str,
    sr: int = 22050,
    n_mels: int = 128,
) -> np.ndarray:
    """Load one audio file and return a 128-dim mel feature vector."""
    y, sr = librosa.load(path, sr=sr, mono=True)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    feat = S.mean(axis=1)  # (128,)
    return feat

def load_dataset_from_folders(base_dir: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load all audio files from
        base_dir/acapella
        base_dir/non_acapella
    and return (X, y, file_paths).
    """
    base = Path(base_dir)
    X = []
    y = []
    files = []

    for label_name, label_id in LABEL_MAP.items():
        folder = base / label_name
        if not folder.exists():
            continue

        for fname in os.listdir(folder):
            if not fname.lower().endswith((".wav", ".mp3", ".m4a", ".flac", ".ogg")):
                continue
            fpath = str(folder / fname)
            feat = extract_mel_features(fpath)
            X.append(feat)
            y.append(label_id)
            files.append(fpath)

    X = np.vstack(X)
    y = np.array(y)
    return X, y, files
