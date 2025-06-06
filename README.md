# Audio Feature Extraction with Librosa

This Python script extracts various audio features from a collection of `.mp3` files using the [Librosa](https://librosa.org/) library. It supports multiprocessing to speed up feature extraction and saves the results into a CSV file.

---

## Features Extracted

- Tempo (beats per minute)
- MFCCs (Mel Frequency Cepstral Coefficients) and their delta features, synchronized to beats
- Chroma features, synchronized to beats
- Spectral centroid
- Spectral bandwidth
- Spectral rolloff
- Zero crossing rate

---

## How It Works

1. Loads each audio file.
2. Computes Short-Time Fourier Transform (STFT) magnitude.
3. Separates harmonic and percussive components.
4. Tracks beats and extracts features synchronized to beats.
5. Aggregates feature statistics (mean values) for each feature.
6. Saves the features along with file identifiers to a CSV file.

The script automatically skips files that have already been processed (based on filenames in the CSV).

---

## Requirements

- Python 3.6+
- librosa
- numpy
- pandas
- tqdm

You can install dependencies using pip:

```bash
pip install librosa numpy pandas tqdm
````

---

## Usage

1. Place your `.mp3` files inside the folder specified by `folder_path` in the script (default: `/home/devsharma/model/song_web/oufile`).
2. Run the script:

```bash
python extract_features.py
```

3. The extracted features will be saved/appended to `song_features.csv` located at `./song_web/`.

---

## Notes

* The script uses multiprocessing to speed up processing. It uses up to 8 CPU cores by default.
* If the output CSV already exists, it will only process new files to avoid duplication.
* Errors during file processing are logged but do not stop the script.
