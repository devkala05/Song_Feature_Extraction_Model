import librosa
import pandas as pd
import os
import numpy as np

folder_path = "/home/devsharma/model/song_web/ooo"
output_file = "./song_web/song_features.csv"

data= []

for file in os.listdir(folder_path):
    if file.endswith(".mp3"):
        try:
            file_path = os.path.join(folder_path, file)
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

            print("data appended")
            data.append(features)
        
        except Exception as e:
            print(f"Error processing {file}: {e}")


df = pd.DataFrame(data)
df.to_csv(output_file, index = False)
