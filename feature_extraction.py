import librosa
import pandas as pd
import os
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import csv

def extract(file_path):
    try:
        y, sr = librosa.load(file_path)
        hop_length = 512
        print("started")
        S, phase = librosa.magphase(librosa.stft(y=y))
        # Harmonic and Percussive separations
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        # Beat tracking
        tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)

        # MFCCs and delta
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
        mfcc_delta = librosa.feature.delta(mfcc)
        beat_mfcc_delta = librosa.util.sync(np.vstack([mfcc, mfcc_delta]), beat_frames)
        # Chroma features
        chromagram = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
        beat_chroma = librosa.util.sync(chromagram, beat_frames, aggregate=np.median)
        # Spectral centroid
        centroid = librosa.feature.spectral_centroid(S=S)
        # Spectral bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(S=S)
        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)
        # Zero crossing rate
        zero_crossing = librosa.feature.zero_crossing_rate(y)
        # Tonnetz
        # y, sr = librosa.load(librosa.ex('nutcracker'), duration=10, offset=10)
        tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)

        file = os.path.basename(file_path)
        features = {
            "id": file.split("_")[0],
            "filename": file,
            "tempo": tempo,
        }
        for i in range(beat_mfcc_delta.shape[0]):
            features[f"mfcc_{i+1}"] = np.mean(beat_mfcc_delta[i])
        for i in range(beat_chroma.shape[0]):
            features[f"chroma_{i+1}"] = np.mean(beat_chroma[i])
        for i in range(centroid.shape[0]):
            features[f"centroid_{i+1}"] = np.mean(centroid[i])
        for i in range(bandwidth.shape[0]):
            features[f"bandwidth_{i+1}"] = np.mean(bandwidth[i])
        for i in range(rolloff.shape[0]):
            features[f"rolloff_{i+1}"] = np.mean(rolloff[i])
        for i in range(zero_crossing.shape[0]):
            features[f"zcr_{i+1}"] = np.mean(zero_crossing[i])

        return features
        
    except Exception as e:
        print(f"Error processing {file}: {e}")
        return None

def write_row_to_csv(row, write_header=False):
    with open(output_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

if __name__ == "__main__":
    folder_path = "/home/devsharma/model/song_web/oufile"
    output_file = "./song_web/song_features.csv"

    if os.path.exists(output_file):
        existss = pd.read_csv(output_file)
        already_processed = set(existss['filename'].tolist())
        write_header = False
    else:
        already_processed = set()
        write_header = True
        
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".mp3")]
    left_files = [f for f in files if os.path.basename(f) not in already_processed]

    print(f"{len(left_files)} files left to process . . .")

    workers = min(cpu_count(), 8)
    
    with Pool(workers) as pool:
        for results in tqdm(pool.imap_unordered(extract, left_files), total = len(left_files)):
            write_row_to_csv(results, write_header)
            write_header = False
            
