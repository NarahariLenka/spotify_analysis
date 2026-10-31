"""
STEP 3 - Exploratory Data Analysis (EDA)
========================================
Run third:   python step3_eda.py      (needs: python step2_preprocessing.py)

Shows each chart on screen (close the window to see the next one) and
prints the key numbers behind it. Each chart is also saved once to
outputs/figures/ so it can be used in the documentation later.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_DIR = Path(__file__).resolve().parent
CLEANED_FILE = PROJECT_DIR / "data" / "processed" / "spotify_tracks_cleaned.csv"
FIGURES_DIR = PROJECT_DIR / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

AUDIO_FEATURES = ["danceability", "energy", "loudness", "valence", "acousticness",
                  "speechiness", "instrumentalness", "liveness", "tempo"]

sns.set_theme(style="whitegrid")


def show_chart(file_name):
    """Save the current chart once, then display it on screen."""
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / file_name, dpi=200, bbox_inches="tight")
    plt.show()


df = pd.read_csv(CLEANED_FILE)
print(f"Working dataset: {len(df):,} tracks, {df['track_genre'].nunique()} genres\n")

# ---------------------------------------------------------------------------
# 1. Popularity distribution
# ---------------------------------------------------------------------------
pop = df["popularity"]
print("1. POPULARITY DISTRIBUTION")
print(pop.describe().round(2).to_string())
print(f"Skewness: {pop.skew():.3f}")
print(f"Tracks with popularity 0: {(pop == 0).sum():,} ({(pop == 0).mean() * 100:.1f}%)")
print("Interpretation: popularity is spread across the whole 0-100 range with a large spike at 0; "
      "most tracks sit between the 25th and 75th percentiles shown above, and very few reach 70+.\n")

plt.figure(figsize=(10, 5))
# one bar per score (bins from -0.5 to 100.5), so the spike at exactly 0 is shown correctly
plt.hist(pop, bins=np.arange(-0.5, 101), color="#4c72b0")
plt.axvline(39.5, color="black", linestyle="--", label="Low / Medium boundary (40)")
plt.axvline(69.5, color="red", linestyle="--", label="Medium / High boundary (70)")
plt.title("Distribution of track popularity")
plt.xlabel("Popularity score (0-100)")
plt.ylabel("Number of tracks")
plt.legend()
show_chart("01_popularity_distribution.png")

# ---------------------------------------------------------------------------
# 2. Duration distribution
# ---------------------------------------------------------------------------
dur = df["duration_min"]
print("2. TRACK DURATION (minutes)")
print(dur.describe().round(2).to_string())
print(f"Skewness: {dur.skew():.2f}   Tracks longer than 10 min: {(dur > 10).sum()}")
print("Interpretation: most tracks last 2-5 minutes; a long right tail of very long tracks "
      "(mixes, ambient pieces) makes the distribution strongly right-skewed. These are real "
      "tracks, so they were kept.\n")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].hist(dur[dur <= 10], bins=60, color="#55a868", edgecolor="white")
axes[0].set_title("Track duration (tracks up to 10 minutes)")
axes[0].set_xlabel("Duration (minutes)")
axes[0].set_ylabel("Number of tracks")
sns.boxplot(x=dur, ax=axes[1], color="#55a868", fliersize=2)
axes[1].set_xscale("log")
axes[1].set_title("Track duration, all tracks (log scale)")
axes[1].set_xlabel("Duration (minutes, log scale)")
show_chart("02_duration_distribution.png")

# ---------------------------------------------------------------------------
# 3. Genre frequency and average popularity by genre (printed; chart in step4_objectives.py)
# ---------------------------------------------------------------------------
genre_counts = df["track_genre"].value_counts()
print("3. GENRE FREQUENCY")
print(f"{len(genre_counts)} genres, between {genre_counts.min()} and {genre_counts.max()} tracks each "
      f"(median {genre_counts.median():.0f}).")
print("Genres with the fewest tracks:", genre_counts.tail(3).to_dict())
print("Interpretation: the original dataset was built with 1,000 tracks per genre, so genre "
      "counts do NOT reflect real-world popularity. A few genres have fewer tracks here because "
      "songs shared with another genre were kept only under the first genre listed.\n")

genre_pop = df.groupby("track_genre")["popularity"].agg(["count", "mean"]).sort_values("mean")
print("Least popular genres (mean popularity):")
print(genre_pop.head(5).round(1).to_string())
print("Most popular genres (mean popularity):")
print(genre_pop.tail(5).round(1).to_string())
print("(Bar chart of genre popularity is shown in step4_objectives.py, Objective 1.)\n")

# ---------------------------------------------------------------------------
# 4. Explicit vs non-explicit (numbers only; chart in step4_objectives.py, Objective 4)
# ---------------------------------------------------------------------------
print("4. EXPLICIT VS NON-EXPLICIT (0 = non-explicit, 1 = explicit)")
print(df.groupby("explicit")["popularity"].agg(["count", "mean", "median"]).round(2).to_string())
print()

# ---------------------------------------------------------------------------
# 5. Popularity vs audio features
# ---------------------------------------------------------------------------
print("5. POPULARITY VS AUDIO FEATURES")
print("Each panel: grey dots = tracks, red line = mean popularity in each tenth (decile) of the feature.")
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
for ax, feature in zip(axes.ravel(), AUDIO_FEATURES):
    data = df[[feature, "popularity"]].dropna()
    ax.scatter(data[feature], data["popularity"], s=3, alpha=0.15, color="grey")
    # split the feature into 10 equal-sized groups and take the mean of each group
    deciles = pd.qcut(data[feature], 10, duplicates="drop")
    decile_means = data.groupby(deciles, observed=True).mean()
    ax.plot(decile_means[feature], decile_means["popularity"], color="red", marker="o",
            linewidth=2, label="Mean popularity per decile")
    unit = {"loudness": "dB", "tempo": "BPM"}.get(feature, "0-1")
    ax.set_title(feature.capitalize())
    ax.set_xlabel(f"{feature} ({unit})")
    ax.set_ylabel("Popularity (0-100)")
axes[0, 0].legend(loc="upper left", fontsize=8)
fig.suptitle("Popularity vs audio features", fontsize=15, fontweight="bold")
show_chart("03_features_vs_popularity.png")
print("Interpretation: the red lines are almost flat - no single audio feature strongly "
      "separates popular from unpopular tracks (quantified in Objective 2).\n")

# ---------------------------------------------------------------------------
# 6. Correlation heatmap
# ---------------------------------------------------------------------------
corr_columns = ["popularity", "duration_min"] + AUDIO_FEATURES + ["explicit", "mode", "num_artists"]
corr = df[corr_columns].corr()
print("6. CORRELATIONS")
print("Correlation of each variable with popularity:")
print(corr["popularity"].drop("popularity").round(3).to_string())

# Find the strongest relationships between two different features
pair_list = []
for i, first in enumerate(corr_columns):
    for second in corr_columns[i + 1:]:
        pair_list.append((first, second, corr.loc[first, second]))
pair_list.sort(key=lambda pair: abs(pair[2]), reverse=True)
print("Strongest correlations between features:")
for first, second, value in pair_list[:3]:
    print(f"  {first} - {second}: {value:.3f}")
print("Interpretation: features are strongly related to EACH OTHER (e.g. energy, loudness and "
      "acousticness), but only weakly related to popularity.\n")

plt.figure(figsize=(11, 9))
# The matrix is symmetric (energy-loudness = loudness-energy), so the upper
# right half would only repeat the lower left half. The mask hides it.
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1,
            annot_kws={"size": 8}, cbar_kws={"label": "Pearson correlation"})
plt.title("Correlation matrix of numerical variables")
plt.xticks(rotation=45, ha="right")
show_chart("04_correlation_heatmap.png")

# ---------------------------------------------------------------------------
# 7. Skewness and outliers
# ---------------------------------------------------------------------------
print("7. SKEWNESS AND OUTLIERS (1.5 x IQR rule)")
for feature in ["duration_min"] + AUDIO_FEATURES:
    values = df[feature].dropna()
    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1
    outliers = ((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum()
    print(f"  {feature:<17} skewness {values.skew():6.2f}   outliers {outliers:>5} "
          f"({outliers / len(values) * 100:.1f}%)")
print("Interpretation: speechiness, instrumentalness, liveness and duration are right-skewed; "
      "their 'outliers' are genuine tracks (e.g. spoken word, instrumentals, live recordings), "
      "so they were kept.\n")

# The seven features measured on the same 0-1 scale can share one chart.
# (Duration outliers are already shown in chart 2.)
features_0_to_1 = ["danceability", "energy", "valence", "acousticness",
                   "speechiness", "instrumentalness", "liveness"]
plt.figure(figsize=(12, 5))
sns.boxplot(data=df[features_0_to_1], color="#64b5cd", fliersize=1)
plt.title("Spread and outliers of the 0-1 audio features (dots = IQR outliers)")
plt.xlabel("Audio feature")
plt.ylabel("Value (0-1)")
show_chart("05_feature_boxplots.png")

# ---------------------------------------------------------------------------
# 8. Popularity category balance
# ---------------------------------------------------------------------------
order = ["Low", "Medium", "High"]
category_counts = df["popularity_category"].value_counts().reindex(order)
print("8. POPULARITY CATEGORY BALANCE")
for category in order:
    print(f"  {category:<7} {category_counts[category]:>5} tracks  ({category_counts[category] / len(df) * 100:.1f}%)")
print("Interpretation: the classes are imbalanced - High is rare - so accuracy alone would be "
      "misleading; macro F1 is used to compare models.\n")

plt.figure(figsize=(7, 5))
bars = plt.bar(order, category_counts.values, color=["#d95f02", "#7570b3", "#1b9e77"])
for bar, count in zip(bars, category_counts.values):
    plt.text(bar.get_x() + bar.get_width() / 2, count, f"{count:,}\n({count / len(df) * 100:.1f}%)",
             ha="center", va="bottom")
plt.ylim(0, category_counts.max() * 1.2)
plt.title("Popularity category balance")
plt.xlabel("Popularity category (Low 0-39, Medium 40-69, High 70-100)")
plt.ylabel("Number of tracks")
show_chart("06_popularity_category_balance.png")

# ---------------------------------------------------------------------------
# 9. Artist and track statistics
# ---------------------------------------------------------------------------
artists = df.groupby("artists")["popularity"].agg(["count", "mean"])
print("9. ARTIST AND TRACK STATISTICS")
print(f"Different artist credits: {len(artists):,}")
print(f"Artist credits with only one track in the sample: {(artists['count'] == 1).sum():,}")
print(f"Tracks with more than one artist: {(df['num_artists'] > 1).mean() * 100:.1f}%")
print("Artists with the most tracks in the sample:")
print(artists.sort_values("count", ascending=False).head(10).round(1).to_string())
print("Most popular tracks in the sample:")
print(df.nlargest(5, "popularity")[["track_name", "artists", "track_genre", "popularity"]].to_string(index=False))
print("Interpretation: most artists appear only once, so artist names would let a model "
      "memorise rather than learn - they are NOT used as model inputs.\n")

print("EDA finished. Next step: python step4_objectives.py")
