"""
STEP 2 - Sampling, cleaning and feature preparation
===================================================
Run second:   python step2_preprocessing.py

1. Takes a reproducible working sample of about 10,000 rows from the raw file.
2. Cleans the sample, printing the row count before and after every step.
3. Creates the target 'popularity_category' and a few simple features.
4. Saves the result to data/processed/spotify_tracks_cleaned.csv
   (the only file this script writes - the raw file is never changed).
"""
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
RAW_FILE = PROJECT_DIR / "data" / "raw" / "dataset.csv"
CLEANED_FILE = PROJECT_DIR / "data" / "processed" / "spotify_tracks_cleaned.csv"

SAMPLE_SIZE = 10_000
RANDOM_STATE = 42


def popularity_to_category(popularity):
    """Low = 0-39, Medium = 40-69, High = 70-100."""
    return pd.cut(popularity, bins=[-1, 39, 69, 100], labels=["Low", "Medium", "High"])


def show_step(step_name, rows_before, rows_after):
    print(f"{step_name:<55} {rows_before:>7,} -> {rows_after:>7,} rows "
          f"(removed {rows_before - rows_after:,})")


if not RAW_FILE.exists():
    RAW_FILE = RAW_FILE.with_name("dataset.csv.zip")
raw = pd.read_csv(RAW_FILE)
print(f"Raw dataset: {raw.shape[0]:,} rows x {raw.shape[1]} columns (left unchanged)")

# ---------------------------------------------------------------------------
# 1. Reproducible stratified sample of ~10,000 rows
# ---------------------------------------------------------------------------
# Every (genre, popularity category) group keeps the same share of rows as in
# the full data, so the genre mix and the Low/Medium/High balance are kept.
fraction = SAMPLE_SIZE / len(raw)
raw["sampling_group"] = raw["track_genre"] + "|" + popularity_to_category(raw["popularity"]).astype(str)

sample_parts = []
for group_name, group_rows in raw.groupby("sampling_group"):
    rows_to_take = round(len(group_rows) * fraction)
    if rows_to_take > 0:
        sample_parts.append(group_rows.sample(n=rows_to_take, random_state=RANDOM_STATE))

df = pd.concat(sample_parts).sort_index().drop(columns="sampling_group")
raw = raw.drop(columns="sampling_group")

print("\n--- Working sample ---")
print(f"Sample size: {len(df):,} rows")
comparison = pd.DataFrame({
    "full data %": (popularity_to_category(raw["popularity"]).value_counts(normalize=True) * 100).round(2),
    "sample %": (popularity_to_category(df["popularity"]).value_counts(normalize=True) * 100).round(2),
})
print(comparison.to_string())
print(f"Tracks per genre in the sample: {df['track_genre'].value_counts().min()} "
      f"to {df['track_genre'].value_counts().max()}")

print("\n--- Sample BEFORE cleaning ---")
print(f"Shape: {df.shape}")
print("Missing values:", int(df.isna().sum().sum()))
print("Exact duplicates (ignoring 'Unnamed: 0'):", int(df.drop(columns="Unnamed: 0").duplicated().sum()))
print("Repeated track_id rows:", int(df["track_id"].duplicated().sum()))

# ---------------------------------------------------------------------------
# 2. Cleaning steps
# ---------------------------------------------------------------------------
print("\n--- Cleaning steps ---")

# Step 1: 'Unnamed: 0' is just the old row number -> drop the column
n = len(df)
df = df.drop(columns=["Unnamed: 0"])
show_step("1. Drop 'Unnamed: 0' (exported row index)", n, len(df))

# Step 2: rows with missing artist / album / track name.
# In the full data there is exactly ONE such row, and it also has a duration
# of 0 ms -> it is a corrupt record, so it is removed instead of filled in.
n = len(df)
df = df.dropna(subset=["artists", "album_name", "track_name"])
show_step("2. Remove corrupt rows with missing names", n, len(df))

# Step 3: tidy text - remove extra spaces, lower-case genre
n = len(df)
genres_before = df["track_genre"].nunique()
for column in ["artists", "album_name", "track_name", "track_genre"]:
    df[column] = df[column].str.strip().str.replace(r"\s+", " ", regex=True)
df["track_genre"] = df["track_genre"].str.lower()
show_step("3. Strip extra spaces in text columns", n, len(df))
print(f"   Genre names before/after tidying: {genres_before} / {df['track_genre'].nunique()} "
      "(equal -> no inconsistent spellings such as 'Pop' vs 'pop')")
print(f"   'explicit' values: {sorted(df['explicit'].unique().tolist())}  "
      f"'mode' values: {sorted(df['mode'].unique().tolist())}")

# Step 4: exact duplicate rows
n = len(df)
df = df.drop_duplicates()
show_step("4. Remove exact duplicate rows", n, len(df))

# Step 5: the same track listed under several genres.
# Audio features are identical for every listing, so we keep the FIRST
# listing only. This stops one song being counted twice in the analysis
# and stops the same song appearing in both the training and test sets.
# Inconsistent data check: does the same track have DIFFERENT values in
# different listings? (Audio features should be identical; popularity can
# differ slightly because the listings were collected at different times.)
popularity_values = df.groupby("track_id")["popularity"].nunique()
energy_values = df.groupby("track_id")["energy"].nunique()
print(f"   Tracks whose listings disagree on popularity: {(popularity_values > 1).sum()}")
print(f"   Tracks whose listings disagree on audio features: {(energy_values > 1).sum()}")
n = len(df)
df = df.drop_duplicates(subset="track_id", keep="first")
show_step("5. Keep one row per track_id (first listing kept)", n, len(df))

# Step 6: tempo = 0 and time_signature = 0 mean "no beat detected"
# (mostly sleep / ambient tracks). They are not real values, so they are set
# to missing. They are filled in later using the TRAINING data only.
tempo_zero = (df["tempo"] == 0).sum()
time_sig_zero = (df["time_signature"] == 0).sum()
df["tempo"] = df["tempo"].replace(0, np.nan)
df["time_signature"] = df["time_signature"].replace(0, np.nan)
print(f"6. tempo = 0 in {tempo_zero} rows and time_signature = 0 in {time_sig_zero} rows "
      "-> set to missing (rows kept)")

# Step 7: validate ranges (the script stops with an error if anything is wrong)
assert df["popularity"].between(0, 100).all()
assert (df["duration_ms"] > 0).all()
assert df["key"].between(0, 11).all() and df["mode"].isin([0, 1]).all()
for column in ["danceability", "energy", "speechiness", "acousticness",
               "instrumentalness", "liveness", "valence"]:
    assert df[column].between(0, 1).all()
print("7. Ranges validated: popularity 0-100, duration > 0, audio features 0-1, key 0-11, mode 0/1")
print("   Unusual but valid values are KEPT: very long tracks, loudness > 0 dB, time_signature 1.")

# ---------------------------------------------------------------------------
# 3. Target and engineered features
# ---------------------------------------------------------------------------
df["popularity_category"] = popularity_to_category(df["popularity"])  # TARGET for ML
df["duration_min"] = df["duration_ms"] / 60000                         # easier to read than ms
df["num_artists"] = df["artists"].str.count(";") + 1                   # collaborations
df["rhythm_undetected"] = df["tempo"].isna().astype(int)               # remembers tempo was 0
df["explicit"] = df["explicit"].astype(int)                            # True/False -> 1/0

print("\n--- Sample AFTER cleaning ---")
print(f"Shape: {df.shape}")
print("Missing values per column (only columns with missing values):")
missing = df.isna().sum()
print(missing[missing > 0].to_string())
print("\nPopularity category counts:")
print(df["popularity_category"].value_counts().to_string())

CLEANED_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(CLEANED_FILE, index=False)
print(f"\nSaved cleaned working dataset -> {CLEANED_FILE.relative_to(PROJECT_DIR)}")
print("Next step: python step3_eda.py")
