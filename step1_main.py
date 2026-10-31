"""
STEP 1 - Dataset inspection
===========================
Run first:   python step1_main.py

Loads the ORIGINAL Spotify file (all ~114,000 rows) and prints:
shape, column names, data types, sample rows, missing values, duplicates,
descriptive statistics, suspicious values and the popularity class balance.

Nothing is changed or saved - this file only looks at the raw data.

Dataset source (verified): "Spotify Tracks Dataset" by Kaggle / Hugging Face
user 'maharshipandya'
  https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
  https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset
  (licence shown on Hugging Face: BSD)
"""
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
RAW_FILE = PROJECT_DIR / "data" / "raw" / "dataset.csv"

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)


def print_title(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


# ---------------------------------------------------------------------------
# 1. Load the raw data (a .csv, or a .zip that contains the csv)
# ---------------------------------------------------------------------------
if not RAW_FILE.exists():
    zipped = RAW_FILE.with_name("dataset.csv.zip")
    if zipped.exists():
        RAW_FILE = zipped
    else:
        raise FileNotFoundError("Put dataset.csv (or dataset.csv.zip) in the data/raw/ folder.")

df = pd.read_csv(RAW_FILE)   # pandas can read a zipped csv directly

print_title("1. SHAPE AND COLUMNS")
print(f"File   : {RAW_FILE.name}")
print(f"Rows   : {df.shape[0]:,}")
print(f"Columns: {df.shape[1]}")
print("\nColumn names and data types:")
print(df.dtypes.to_string())

print_title("2. FIRST FIVE ROWS")
print(df.head())

# ---------------------------------------------------------------------------
# 2. Missing values
# ---------------------------------------------------------------------------
print_title("3. MISSING VALUES")
missing = pd.DataFrame({
    "missing_count": df.isna().sum(),
    "missing_percent": (df.isna().mean() * 100).round(4),
})
print(missing.to_string())
print("\nRows that contain a missing value:")
print(df[df.isna().any(axis=1)].T)

# ---------------------------------------------------------------------------
# 3. Duplicates
# ---------------------------------------------------------------------------
print_title("4. DUPLICATES")
# 'Unnamed: 0' is unique on every row, so we also check WITHOUT it
without_index = df.drop(columns=["Unnamed: 0"])
print(f"Exact duplicate rows (all columns)            : {df.duplicated().sum()}")
print(f"Exact duplicate rows (ignoring 'Unnamed: 0')   : {without_index.duplicated().sum()}")
print(f"Rows whose track_id appeared earlier          : {df['track_id'].duplicated().sum():,}")
print(f"Unique track_id values                        : {df['track_id'].nunique():,}")
genres_per_track = df.groupby("track_id")["track_genre"].nunique()
print(f"Tracks listed under more than one genre       : {(genres_per_track > 1).sum():,}")
print("-> The same song can appear once for every genre it belongs to.")
popularity_per_track = df.groupby("track_id")["popularity"].nunique()
print(f"Tracks whose listings have DIFFERENT popularity    : {(popularity_per_track > 1).sum()}")
print("-> Inconsistent data: handled in step2_preprocessing.py by keeping the first listing.")

is_row_number = (df["Unnamed: 0"] == range(len(df))).all()
print(f"\n'Unnamed: 0' equals the row number 0..{len(df) - 1}: {is_row_number}")
print("-> It is an index column saved by mistake when the csv was exported.")

# ---------------------------------------------------------------------------
# 4. Unique values / categorical columns
# ---------------------------------------------------------------------------
print_title("5. UNIQUE VALUES PER COLUMN")
print(df.nunique().to_string())
print("\nTracks per genre (first 5 genres):")
print(df["track_genre"].value_counts().head().to_string())
print(f"Number of genres: {df['track_genre'].nunique()}")
print("\nExplicit:", df["explicit"].value_counts().to_dict())
print("Mode    :", df["mode"].value_counts().to_dict())
print("Time signature:", df["time_signature"].value_counts().sort_index().to_dict())

# ---------------------------------------------------------------------------
# 5. Descriptive statistics
# ---------------------------------------------------------------------------
print_title("6. DESCRIPTIVE STATISTICS (numerical columns)")
print(df.describe().T.round(3).to_string())

# ---------------------------------------------------------------------------
# 6. Suspicious / invalid values
# ---------------------------------------------------------------------------
print_title("7. SUSPICIOUS OR INVALID VALUES")
print(f"popularity outside 0-100        : {(~df['popularity'].between(0, 100)).sum()}")
print(f"popularity equal to 0           : {(df['popularity'] == 0).sum():,}")
print(f"duration_ms equal to 0          : {(df['duration_ms'] == 0).sum()}")
print(f"tracks longer than 20 minutes   : {(df['duration_ms'] > 20 * 60000).sum()}")
print(f"tempo equal to 0 BPM            : {(df['tempo'] == 0).sum()}")
print(f"time_signature equal to 0       : {(df['time_signature'] == 0).sum()}")
print(f"loudness above 0 dB             : {(df['loudness'] > 0).sum()}")
print(f"key outside 0-11                : {(~df['key'].between(0, 11)).sum()}")
for column in ["danceability", "energy", "speechiness", "acousticness",
               "instrumentalness", "liveness", "valence"]:
    print(f"{column:<17} outside 0-1    : {(~df[column].between(0, 1)).sum()}")
artist_text = df["artists"].fillna("")
print(f"artists with extra spaces       : {(artist_text != artist_text.str.strip()).sum()}")

# Potential outliers using the 1.5 x IQR rule
print("\nPotential outliers (1.5 x IQR rule) and skewness:")
for column in ["duration_ms", "loudness", "speechiness", "instrumentalness", "liveness", "tempo"]:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    n_outliers = ((df[column] < q1 - 1.5 * iqr) | (df[column] > q3 + 1.5 * iqr)).sum()
    print(f"  {column:<17} outliers: {n_outliers:>6,}   skewness: {df[column].skew():6.2f}")

# ---------------------------------------------------------------------------
# 7. Popularity class balance (thresholds used later for the ML target)
# ---------------------------------------------------------------------------
print_title("8. POPULARITY CATEGORY BALANCE (Low 0-39, Medium 40-69, High 70-100)")
category = pd.cut(df["popularity"], bins=[-1, 39, 69, 100], labels=["Low", "Medium", "High"])
balance = pd.DataFrame({
    "tracks": category.value_counts(),
    "percent": (category.value_counts(normalize=True) * 100).round(2),
})
print(balance.to_string())
print("\nThe High class is rare (about 5%), so macro F1 will be an important metric.")
print("\nInspection finished. Next step: python step2_preprocessing.py")
