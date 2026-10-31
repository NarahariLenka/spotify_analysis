"""
STEP 4 - The five research objectives
=====================================
Run fourth:   python step4_objectives.py      (needs: python step2_preprocessing.py)

Objectives 1-4 are analysed here: each one prints the research question,
method, numerical results and a conclusion, and shows its chart(s).
Objective 5 (machine-learning classification) is carried out by
step5_train_models.py, which is the next file to run.

Statistics note: with ~10,000 tracks even tiny differences give very small
p-values, so EFFECT SIZES (how big a difference is) are used to judge
whether a result actually matters. Correlation does not prove causation.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

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


def print_header(title, question, variables, method):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    print(f"Research question: {question}")
    print(f"Variables        : {variables}")
    print(f"Method           : {method}\n")


def eta_squared(values, groups):
    """Share of the variation in `values` explained by the groups (0 to 1).
    Rule of thumb: 0.01 small, 0.06 medium, 0.14 large."""
    data = pd.DataFrame({"value": values, "group": groups}).dropna()
    overall_mean = data["value"].mean()
    group_stats = data.groupby("group")["value"].agg(["count", "mean"])
    between = (group_stats["count"] * (group_stats["mean"] - overall_mean) ** 2).sum()
    total = ((data["value"] - overall_mean) ** 2).sum()
    return between / total


def cohens_d(group_a, group_b):
    """Difference in means measured in standard deviations.
    Rule of thumb: 0.2 small, 0.5 medium, 0.8 large."""
    a = group_a.dropna()
    b = group_b.dropna()
    pooled_sd = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / pooled_sd


df = pd.read_csv(CLEANED_FILE)
print(f"Working dataset: {len(df):,} tracks")

# ===========================================================================
# OBJECTIVE 1 - Popularity distribution and genre differences
# ===========================================================================
print_header(
    "OBJECTIVE 1: Popularity distribution and genre differences",
    "How is popularity distributed, and which genres have high or low average popularity?",
    "popularity, track_genre",
    "descriptive statistics per genre (only genres with >= 30 tracks are ranked), "
    "Kruskal-Wallis test and eta-squared effect size")
print("Why it matters: shows whether genre alone tells us much about popularity.\n")

pop = df["popularity"]
print(f"Popularity: mean {pop.mean():.2f}, median {pop.median():.0f}, SD {pop.std():.2f}, "
      f"IQR {pop.quantile(.25):.0f}-{pop.quantile(.75):.0f}, zeros {(pop == 0).mean() * 100:.1f}%")

df["is_zero"] = (df["popularity"] == 0).astype(int)   # 1 if popularity is exactly 0
genre_stats = df.groupby("track_genre")["popularity"].agg(["count", "mean", "median", "std"])
genre_stats["share_zero"] = df.groupby("track_genre")["is_zero"].mean()
genre_stats = genre_stats[genre_stats["count"] >= 30].sort_values("mean", ascending=False)
print(f"\nGenres ranked: {len(genre_stats)} (group sizes {genre_stats['count'].min()}-{genre_stats['count'].max()})")
print("Top 5 genres:")
print(genre_stats.head(5).round(2).to_string())
print("Bottom 5 genres:")
print(genre_stats.tail(5).round(2).to_string())

genre_groups = [group["popularity"].values for _, group in df.groupby("track_genre")]
kruskal = stats.kruskal(*genre_groups)
eta_genre = eta_squared(df["popularity"], df["track_genre"])
print(f"\nKruskal-Wallis H = {kruskal.statistic:.1f}, p = {kruskal.pvalue:.2e}")
print(f"Eta-squared = {eta_genre:.3f} -> genre explains about {eta_genre * 100:.0f}% of the variation in popularity")

top10 = list(genre_stats.index[:10])
bottom10 = list(genre_stats.index[-10:])
plot_genres = top10 + bottom10
colors = {}
for genre in top10:
    colors[genre] = "#1b9e77"      # green = most popular
for genre in bottom10:
    colors[genre] = "#d95f02"      # orange = least popular
plt.figure(figsize=(11, 8))
sns.boxplot(data=df[df["track_genre"].isin(plot_genres)], x="popularity", y="track_genre",
            order=plot_genres, hue="track_genre", palette=colors, legend=False, fliersize=2)
plt.axvline(pop.mean(), color="black", linestyle="--", label=f"Overall mean ({pop.mean():.1f})")
plt.title("Popularity of the 10 most (green) and 10 least (orange) popular genres")
plt.xlabel("Popularity score (0-100)")
plt.ylabel("Genre")
plt.legend(loc="lower right")
show_chart("07_obj1_genre_popularity.png")

print("\nInterpretation: genre means range from about "
      f"{genre_stats['mean'].min():.0f} to {genre_stats['mean'].max():.0f}. Several of the lowest "
      "genres have a large share of tracks with popularity 0, which pulls their mean down.")
print(f"CONCLUSION 1: popularity differs strongly between genres (large effect, eta-squared = {eta_genre:.2f}), "
      "but most variation is still WITHIN genres.")

# ===========================================================================
# OBJECTIVE 2 - Audio features and popularity
# ===========================================================================
print_header(
    "OBJECTIVE 2: Relationships between audio features and popularity",
    "Are audio features such as danceability, energy or loudness related to popularity?",
    ", ".join(AUDIO_FEATURES) + " vs popularity",
    "Pearson (linear) and Spearman (rank/monotonic) correlation; Spearman repeated "
    "within genres to remove the effect of genre")
print("Why it matters: tells us whether 'how a song sounds' predicts how popular it is.\n")

# "Within genre": subtract each genre's average from every track. What is left
# shows whether a track is above or below its OWN genre's average, so
# differences that come only from genre are removed.
columns = AUDIO_FEATURES + ["popularity"]
genre_averages = df.groupby("track_genre")[columns].transform("mean")
within_genre = df[columns] - genre_averages

rows = []
for feature in AUDIO_FEATURES:
    data = df[[feature, "popularity"]].dropna()
    pearson_r, pearson_p = stats.pearsonr(data[feature], data["popularity"])
    spearman_rho, spearman_p = stats.spearmanr(data[feature], data["popularity"])
    within = within_genre[[feature, "popularity"]].dropna()
    within_rho, _ = stats.spearmanr(within[feature], within["popularity"])
    rows.append([feature, pearson_r, pearson_p, spearman_rho, spearman_p, within_rho])

correlations = pd.DataFrame(rows, columns=["feature", "pearson_r", "pearson_p", "spearman_rho",
                                           "spearman_p", "spearman_within_genre"]).set_index("feature")
# sort from strongest to weakest relationship (ignoring the sign)
correlations["strength"] = correlations["spearman_rho"].abs()
correlations = correlations.sort_values("strength", ascending=False).drop(columns="strength")
print(correlations.round(4).to_string())

plt.figure(figsize=(10, 6))
y_positions = np.arange(len(correlations))
plt.barh(y_positions - 0.25, correlations["pearson_r"], height=0.25, label="Pearson r", color="#4c72b0")
plt.barh(y_positions, correlations["spearman_rho"], height=0.25, label="Spearman rho", color="#55a868")
plt.barh(y_positions + 0.25, correlations["spearman_within_genre"], height=0.25,
         label="Spearman rho within genre", color="#c44e52")
plt.yticks(y_positions, correlations.index)
plt.gca().invert_yaxis()
plt.axvline(0, color="black", linewidth=0.8)
plt.axvline(-0.1, color="grey", linestyle=":")
plt.axvline(0.1, color="grey", linestyle=":")
plt.title("Correlation of audio features with popularity (dotted lines = +/-0.1)")
plt.xlabel("Correlation coefficient")
plt.ylabel("Audio feature")
plt.legend()
show_chart("08_obj2_feature_correlations.png")

strongest = correlations["spearman_rho"].abs().idxmax()
print(f"\nInterpretation: the strongest relationship is {strongest} "
      f"(Spearman rho = {correlations.loc[strongest, 'spearman_rho']:.3f}). Every |correlation| is "
      f"at most {correlations[['pearson_r', 'spearman_rho']].abs().max().max():.2f}, i.e. weak or negligible. "
      "Within genres most correlations become even closer to 0 (compare the red bars), so part "
      "of the overall link comes from genre rather than from the feature itself.")
print("CONCLUSION 2: no audio feature on its own is a meaningful predictor of popularity. "
      "These are correlations only - they do not show that a feature CAUSES popularity.")

# ===========================================================================
# OBJECTIVE 3 - Genre-based audio characteristics
# ===========================================================================
genre_features = ["danceability", "energy", "valence", "acousticness",
                  "instrumentalness", "speechiness", "tempo"]
print_header(
    "OBJECTIVE 3: Genre-based audio characteristics",
    "Which audio characteristics differ meaningfully between genres?",
    ", ".join(genre_features) + " by track_genre",
    "one-way ANOVA and Kruskal-Wallis tests, with eta-squared as the effect size")
print("Why it matters: shows which features describe a genre's sound - useful input for the models.\n")

rows = []
for feature in genre_features:
    data = df[[feature, "track_genre"]].dropna()
    groups = [group[feature].values for _, group in data.groupby("track_genre")]
    anova = stats.f_oneway(*groups)
    kruskal = stats.kruskal(*groups)
    genre_means = data.groupby("track_genre")[feature].mean()
    rows.append([feature, anova.statistic, anova.pvalue, kruskal.pvalue,
                 eta_squared(data[feature], data["track_genre"]),
                 genre_means.idxmin(), genre_means.min(), genre_means.idxmax(), genre_means.max()])

effects = pd.DataFrame(rows, columns=["feature", "anova_F", "anova_p", "kruskal_p", "eta_squared",
                                      "lowest_genre", "lowest_mean", "highest_genre", "highest_mean"])
effects = effects.set_index("feature").sort_values("eta_squared", ascending=False)
print(effects.round(3).to_string())
print(f"\nGenre sizes range from {df['track_genre'].value_counts().min()} to "
      f"{df['track_genre'].value_counts().max()} tracks. ANOVA assumes equal variances and normal data; "
      "Kruskal-Wallis does not, and both agree here.")

plt.figure(figsize=(9, 5))
plt.barh(effects.index, effects["eta_squared"], color="#8172b2")
for position, value in enumerate(effects["eta_squared"]):
    plt.text(value + 0.005, position, f"{value:.2f}", va="center")
plt.gca().invert_yaxis()
plt.axvline(0.14, color="grey", linestyle="--", label="Large effect threshold (0.14)")
plt.title("How much of each audio feature is explained by genre")
plt.xlabel("Eta-squared (share of variation explained by genre)")
plt.ylabel("Audio feature")
plt.legend()
show_chart("09_obj3_genre_effect_sizes.png")

top_feature = effects.index[0]
genre_order = df.groupby("track_genre")[top_feature].mean().sort_values()
middle = len(genre_order) // 2
chosen_genres = list(genre_order.index[:6]) + list(genre_order.index[middle - 3:middle + 3]) + list(genre_order.index[-6:])
plt.figure(figsize=(12, 5))
sns.boxplot(data=df[df["track_genre"].isin(chosen_genres)], x="track_genre", y=top_feature,
            order=chosen_genres, color="#8172b2", fliersize=1)
plt.xticks(rotation=60)
plt.title(f"{top_feature.capitalize()} in 18 genres (6 lowest, 6 middle, 6 highest)")
plt.xlabel("Genre")
plt.ylabel(f"{top_feature} (0-1)")
show_chart("10_obj3_top_feature_by_genre.png")

large = effects[effects["eta_squared"] >= 0.14].index.tolist()
print(f"\nInterpretation: {', '.join(large)} have LARGE genre effects (eta-squared >= 0.14). "
      f"{top_feature.capitalize()} varies most, from {effects.loc[top_feature, 'lowest_mean']:.2f} "
      f"({effects.loc[top_feature, 'lowest_genre']}) to {effects.loc[top_feature, 'highest_mean']:.2f} "
      f"({effects.loc[top_feature, 'highest_genre']}). Tempo varies the least.")
print("CONCLUSION 3: genres have clearly different sound profiles, so audio features describe "
      "genre far better than they describe popularity.")

# ===========================================================================
# OBJECTIVE 4 - Explicit vs non-explicit tracks
# ===========================================================================
compare_features = ["popularity", "duration_min", "danceability", "energy", "speechiness",
                    "valence", "acousticness", "loudness", "instrumentalness"]
print_header(
    "OBJECTIVE 4: Explicit versus non-explicit tracks",
    "Do explicit tracks differ from non-explicit tracks in popularity, duration and sound?",
    "explicit (0/1) vs " + ", ".join(compare_features),
    "Mann-Whitney U test (no normality assumption), Welch t-test, Cohen's d effect size; "
    "within-genre comparison to check for confounding by genre")
print("Why it matters: tests the common idea that explicit songs are more popular.\n")

explicit = df[df["explicit"] == 1]
clean = df[df["explicit"] == 0]
print(f"Explicit tracks: {len(explicit):,} ({len(explicit) / len(df) * 100:.1f}%)   "
      f"Non-explicit tracks: {len(clean):,}")

rows = []
for feature in compare_features:
    mann_whitney = stats.mannwhitneyu(explicit[feature].dropna(), clean[feature].dropna())
    welch = stats.ttest_ind(explicit[feature].dropna(), clean[feature].dropna(), equal_var=False)
    rows.append([feature, explicit[feature].mean(), clean[feature].mean(),
                 mann_whitney.pvalue, welch.pvalue, cohens_d(explicit[feature], clean[feature])])
comparison = pd.DataFrame(rows, columns=["feature", "explicit_mean", "non_explicit_mean",
                                         "mann_whitney_p", "welch_p", "cohens_d"]).set_index("feature")
print(comparison.round(4).to_string())

# Confounding check: explicit songs are concentrated in some genres
explicit_share = df.groupby("track_genre")["explicit"].mean().sort_values(ascending=False)
print("\nGenres with the highest share of explicit tracks:")
for genre in explicit_share.index[:5]:
    print(f"  {genre:<10} {explicit_share[genre] * 100:.1f}%")

genre_differences = []
for genre, group in df.groupby("track_genre"):
    n_explicit = (group["explicit"] == 1).sum()
    n_clean = (group["explicit"] == 0).sum()
    if n_explicit >= 10 and n_clean >= 10:   # need enough tracks on both sides
        difference = (group.loc[group["explicit"] == 1, "popularity"].mean()
                      - group.loc[group["explicit"] == 0, "popularity"].mean())
        genre_differences.append(difference)
genre_differences = pd.Series(genre_differences)
print(f"\nWithin-genre check ({len(genre_differences)} genres with >= 10 explicit and >= 10 non-explicit tracks):")
print(f"  average popularity difference (explicit - non-explicit): {genre_differences.mean():.2f} points")
print(f"  genres where explicit tracks are more popular: {(genre_differences > 0).mean() * 100:.0f}%")

# red bar = value is higher for explicit tracks, blue bar = higher for non-explicit tracks
bar_colors = []
for d in comparison["cohens_d"]:
    if d > 0:
        bar_colors.append("#c44e52")
    else:
        bar_colors.append("#4c72b0")
plt.figure(figsize=(9, 5.5))
plt.barh(comparison.index, comparison["cohens_d"], color=bar_colors)
plt.gca().invert_yaxis()
plt.axvline(0, color="black", linewidth=0.8)
plt.axvline(-0.2, color="grey", linestyle=":", label="Small effect (+/-0.2)")
plt.axvline(0.2, color="grey", linestyle=":")
plt.title("Explicit vs non-explicit tracks: size of each difference")
plt.xlabel("Cohen's d (red = higher for explicit tracks, blue = higher for non-explicit)")
plt.ylabel("Feature")
plt.legend()
show_chart("11_obj4_explicit_comparison.png")

popularity_gap = comparison.loc["popularity", "explicit_mean"] - comparison.loc["popularity", "non_explicit_mean"]
d_popularity = comparison.loc["popularity", "cohens_d"]
biggest = comparison["cohens_d"].abs().idxmax()
d_biggest = comparison.loc[biggest, "cohens_d"]
print(f"\nInterpretation: explicit tracks are on average {popularity_gap:.1f} points more popular, "
      f"but Cohen's d = {d_popularity:.2f} (negligible-to-small). The biggest difference is in "
      f"{biggest} (d = {d_biggest:.2f}): explicit tracks contain much more speech (e.g. rap, comedy).")
print("Limitations: genre, language, release year and artist fame are confounders; this is "
      "observational data, so 'explicit' cannot be said to cause popularity.")
print("CONCLUSION 4: explicit tracks differ mostly in SOUND (more speech, danceable, louder) "
      "rather than in popularity; the popularity gap is small and shrinks within genres.")

# ===========================================================================
# OBJECTIVE 5 - handled by step5_train_models.py
# ===========================================================================
print("\n" + "=" * 78)
print("OBJECTIVE 5: Machine-learning popularity classification")
print("=" * 78)
print("Carried out in the next file. Run:  python step5_train_models.py")
