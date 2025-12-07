import os
import numpy as np
from preprocessing.audio import load_and_process_audio

def build_dataset(raw_dir="data/raw", processed_dir="data/processed", sr=22050, n_mels=128):
    X = []
    y = []

    for label, folder in enumerate(["acapella", "non_acapella"]):
        folder_path = os.path.join(raw_dir, folder)
        for filename in os.listdir(folder_path):
            if filename.lower().endswith((".wav", ".mp3", ".m4a", ".flac")):
                file_path = os.path.join(folder_path, filename)

                # Process the audio
                S_db = load_and_process_audio(file_path, sr=sr, n_mels=n_mels)

                # Save processed file
                out_name = f"{folder}_{os.path.splitext(filename)[0]}_mel.npy"
                out_path = os.path.join(processed_dir, out_name)
                np.save(out_path, S_db)

                # Append to dataset
                X.append(S_db)
                y.append(label)

    return np.array(X, dtype=object), np.array(y)
