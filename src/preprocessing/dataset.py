import os
import numpy as np
from preprocessing.audio import load_and_process_audio

def build_dataset(raw_dir="data/raw", processed_dir="data/processed", sr=22050, n_mels=128):
    """
    Build a dataset from all audio files in data/raw/acapella and data/raw/non_acapella.

    Returns:
        X: list of 2D spectrogram arrays (each (n_mels, T))
        y: 1D numpy array of integer labels (1 for acapella, 0 for non-acapella)
    """
    X = []
    y = []

    # map folder name -> label
    class_map = {
        "acapella": 1,
        "non_acapella": 0,
    }

    os.makedirs(processed_dir, exist_ok=True)

    for folder, label in class_map.items():
        folder_path = os.path.join(raw_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        for filename in os.listdir(folder_path):
            if not filename.lower().endswith((".wav", ".mp3", ".m4a", ".flac")):
                continue

            file_path = os.path.join(folder_path, filename)

            # Compute mel spectrogram
            S_db = load_and_process_audio(file_path, sr=sr, n_mels=n_mels)

            # Save processed file for reproducibility
            name_root, _ = os.path.splitext(filename)
            out_name = f"{folder}_{name_root}_mel.npy"
            out_path = os.path.join(processed_dir, out_name)
            np.save(out_path, S_db)

            # Add to dataset
            X.append(S_db)     # spectrogram
            y.append(label)    # integer label

    # NOTE: X stays a plain Python list of 2D arrays
    return X, np.array(y, dtype=np.int64)
