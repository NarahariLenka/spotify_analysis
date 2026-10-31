# Spotify Music Analytics: Exploratory Data Analysis and Machine Learning-Based Popularity Classification

A student data-science project. It cleans and explores a Spotify tracks dataset, investigates five research questions, and compares five classic classifiers (Logistic Regression, Decision Tree, Random Forest, KNN and Gaussian Naive Bayes). The classifiers predict whether a track's popularity is **Low**, **Medium** or **High**.

The project is five Python files that you run in order. Each one prints its results in the terminal and opens its charts on screen.

---

## Problem statement

Spotify gives every track a popularity score from 0 to 100 and a set of audio features (danceability, energy, loudness and so on). This project asks two things:

- How is popularity distributed across genres and audio characteristics?
- Can simple machine-learning models predict a track's popularity *category* from its genre and audio features alone?

## Five research objectives

| # | Research question | Answered in |
|---|---|---|
| 1 | How is popularity distributed, and which genres have high or low average popularity? | `step4_objectives.py` |
| 2 | Are audio features related to popularity? | `step4_objectives.py` |
| 3 | Which audio characteristics differ meaningfully between genres? | `step4_objectives.py` |
| 4 | Do explicit and non-explicit tracks differ in popularity, duration and sound? | `step4_objectives.py` |
| 5 | How well can the five classifiers predict Low / Medium / High popularity, and which features help? | `step5_train_models.py` |

## Dataset

- **Name:** Spotify Tracks Dataset (file `dataset.csv`, 114,000 rows × 21 columns, verified in `step1_main.py`).
- **Source:** published on Kaggle and Hugging Face by the user **maharshipandya**:
  - https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
  - https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset
- **Licence:** Hugging Face lists the licence as **BSD**. I could not confirm a licence on the Kaggle page.
- **Columns:** `Unnamed: 0, track_id, artists, album_name, track_name, popularity, duration_ms, explicit, danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, time_signature, track_genre`.

The raw file in `data/raw/` is never modified (its SHA-256 checksum was checked before and after processing).

## Technologies

Python 3.12, pandas, NumPy, Matplotlib, Seaborn, SciPy and scikit-learn (exact versions are in `requirements.txt`).

## Project structure

```
spotify_project_sem5/
├── step1_main.py           # STEP 1  inspect the raw dataset (prints only)
├── step2_preprocessing.py  # STEP 2  sample ~10,000 rows, clean, create target and features
├── step3_eda.py            # STEP 3  exploratory charts + key statistics
├── step4_objectives.py     # STEP 4  research objectives 1-4 (tests, charts, conclusions)
├── step5_train_models.py   # STEP 5  objective 5: five classifiers, comparison, best model
├── requirements.txt
├── .gitignore
├── README.md
├── data/
│   ├── raw/dataset.csv                          # original data, unchanged
│   └── processed/spotify_tracks_cleaned.csv     # cleaned working dataset
├── models/best_model.joblib                     # best model + its fitted preprocessing
├── outputs/
│   ├── model_comparison.csv                     # final metrics of all five models
│   └── figures/                                 # the 18 charts shown during execution
└── screenshots/  (vscode/, visualizations/, linkedin/, github/)
```

## Installation and execution (VS Code on Windows)

1. Open the folder in VS Code: **File → Open Folder… → `spotify_project_sem5`**.
2. Open a terminal: **Terminal → New Terminal** (PowerShell).
3. *(Optional but recommended)* Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
   Then choose this interpreter with **Ctrl+Shift+P → "Python: Select Interpreter" → .venv**.
4. Install the libraries:
   ```powershell
   pip install -r requirements.txt
   ```
5. Run the files **in this order**. Either type the commands in the terminal, or open each file and click the ▶ **Run Python File** button at the top right:

   | Order | Command | What you will see | Approx. time |
   |---|---|---|---|
   | 1 | `python step1_main.py` | Shape, columns, types, missing values, duplicates and statistics of the full raw data | ~5 s |
   | 2 | `python step2_preprocessing.py` | Sampling, each cleaning step with row counts before and after, and the target classes | ~5 s |
   | 3 | `python step3_eda.py` | 6 charts plus printed statistics and interpretations | ~10 s + viewing |
   | 4 | `python step4_objectives.py` | Objectives 1–4: statistics, 5 charts and conclusions | ~10 s + viewing |
   | 5 | `python step5_train_models.py` | Each model in turn (accuracy, confusion matrix, report), 7 charts, best model, recommendations | ~10 s + viewing |

   **Each chart opens in its own window. Close the window to continue to the next one.**

   To run everything in one go (PowerShell):
   ```powershell
   python step1_main.py; python step2_preprocessing.py; python step3_eda.py; python step4_objectives.py; python step5_train_models.py
   ```

Files 3–5 only need `step2_preprocessing.py` to have been run once, because they read `data/processed/spotify_tracks_cleaned.csv`.

## Preprocessing methodology (`step2_preprocessing.py`)

1. **Working sample (reproducible).** About 10,000 rows (9,997) are sampled with `random_state=42`. The sample is stratified by *genre × popularity category*, so both are represented as in the full data (Low 57.4 / Medium 37.8 / High 4.8 % in both).
2. **`Unnamed: 0` dropped.** The column is identical to the row number 0…113,999, so it is a saved index.
3. **Corrupt record rule.** In the full data, one row has no artist, album or track name *and* a duration of 0 ms. Rows like this are removed rather than filled in. That row is not in the sample, so 0 rows were removed.
4. **Text tidied.** Extra spaces are stripped and genre is lower-cased. A check confirms there are no inconsistent genre spellings (114 genres before and after tidying).
5. **Exact duplicate rows removed:** 3.
6. **One row per track (inconsistent data).** 291 rows were the same song listed under another genre (same `track_id`, identical audio features). 13 of these tracks had *conflicting* popularity values between listings (720 in the full data). Only the first listing is kept. This prevents a song from being counted twice, or from appearing in both the training and the test set.
7. **Undetected rhythm.** `tempo = 0` (10 rows) and `time_signature = 0` (11 rows) mean "no beat detected", so they are set to missing. They are filled in later with training-data values only.
8. **Ranges validated.** Popularity is 0–100, duration > 0, audio features 0–1, key 0–11 and mode 0/1. Unusual but genuine values (tracks over 20 minutes, loudness > 0 dB, skewed speechiness and instrumentalness) are **kept**, because they are real tracks.

**Result:** 9,703 tracks × 24 columns, saved to `data/processed/spotify_tracks_cleaned.csv`.

### Target and engineered features

| Column | Why |
|---|---|
| `popularity_category` | **Target only.** Low = 0–39, Medium = 40–69, High = 70–100 |
| `duration_min` | `duration_ms / 60000`, easier to read |
| `num_artists` | Number of credited artists (collaborations) without using high-cardinality names |
| `rhythm_undetected` | 1 if tempo was 0, so this information isn't lost after imputation |

**Threshold limitations:** the cut-offs (40 and 70) are fixed choices, not natural boundaries. A track scoring 39 and one scoring 40 are almost identical but fall into different classes. High is rare (4.6 % of tracks). The models predict a *category*, not the exact popularity score.

## EDA findings (`step3_eda.py`, 9,703 tracks)

- **Popularity:** mean 33.3, median 34, SD 22.2. 13.8 % of tracks have popularity 0.
- **Duration:** median 3.6 minutes. The distribution is strongly right-skewed (skewness 10.7), with 52 tracks longer than 10 minutes.
- **Genres:** 114 genres. The dataset was built with 1,000 tracks per genre, so genre counts do not reflect real-world popularity.
- **Correlations:** audio features are strongly related to each other (energy–loudness r = 0.76, energy–acousticness r = −0.74), but only weakly related to popularity (largest is instrumentalness, r = −0.10).
- **Class balance:** Low 57.7 %, Medium 37.8 %, High 4.6 %. Because the classes are imbalanced, macro F1 is the main metric.

## Results of the research objectives

**Objective 1 – Genre differences.** Genre explains about 26 % of the variation in popularity (eta-squared = 0.26, a large effect; Kruskal–Wallis p < 0.001).
- Highest-mean genres: pop-film (59.8), k-pop (56.9), chill (53.8).
- Lowest-mean genres: iranian (2.0), romance (2.7), latin (8.4). Several of the lowest genres have many zero-popularity tracks (iranian 73 %, jazz 73 %, romance 60 %, latin 50 %).
- Every ranked genre has at least 44 tracks.

**Objective 2 – Audio features vs popularity.** Every correlation is weak or negligible (|r| ≤ 0.10). The strongest is instrumentalness (Spearman ρ = −0.08). Most correlations move even closer to zero within genres. *Correlation does not imply causation.*

**Objective 3 – Genre sound profiles.** Six of the seven features have *large* genre effects:
- acousticness: eta² = 0.50 (black-metal 0.01 vs classical 0.90)
- energy: 0.46
- instrumentalness: 0.46
- speechiness: 0.44
- danceability: 0.43
- valence: 0.31
- tempo: 0.09 (medium effect)

Audio features describe genre much better than they describe popularity.

**Objective 4 – Explicit vs non-explicit.** 8.2 % of tracks are explicit.
- **Popularity:** explicit tracks score 3.7 points higher on average (36.6 vs 33.0). This is a negligible effect (Cohen's d = 0.16).
- **Within genres:** in 27 genres with enough explicit and non-explicit tracks, the gap shrinks to 1.5 points.
- **Sound:** the big difference is speechiness (d = 1.28). Explicit tracks are also more danceable, louder and less instrumental.
- **Confounders:** genre, language, artist fame and release year all affect these comparisons.

**Objective 5 – Classification.** See the next section.

## Machine-learning models (`step5_train_models.py`)

The code is written step by step: each model has its own block that creates it, trains it (`fit`), predicts the test data (`predict`), prints the accuracy, confusion matrix and classification report, and shows its confusion-matrix chart.

**Setup**
- **Inputs (17):** 11 numeric features, 3 binary features, and genre, key and time signature (one-hot encoded with `pd.get_dummies`), giving 144 columns in total.
- **Excluded inputs:** `popularity`, `popularity_category`, `track_id`, and the artist, album and track names.
- **Split:** 80 / 20 stratified, `random_state=42`, giving 7,762 training and 1,941 test tracks. All five models use the same split.

**Leakage prevention**
- The split is made before any preparation of the data.
- The missing tempo and time signature values are filled with the *training* median and mode.
- The test data gets exactly the training data's one-hot columns.
- The `StandardScaler` is fitted on the training data only (`fit_transform` on train, `transform` on test).

**Model settings**

| Model | Settings |
|---|---|
| `LogisticRegression` | `C=0.1, class_weight="balanced", max_iter=3000` |
| `DecisionTreeClassifier` | default (grown fully) |
| `RandomForestClassifier` | `n_estimators=200, min_samples_leaf=5, class_weight="balanced"` |
| `KNeighborsClassifier` | `n_neighbors=5` (needs scaled inputs) |
| `GaussianNB` | `var_smoothing=1e-3` |

`class_weight="balanced"` gives the rare High class more weight. GaussianNB assumes each feature is normally distributed and independent of the others within a class. One-hot and skewed features break this assumption, so it serves as a simple baseline.

### Results (test set, 1,941 tracks; precision and recall are macro averages)

| Model | Accuracy | Precision | Recall | **Macro F1** | Weighted F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.602 | 0.525 | 0.601 | **0.507** | 0.642 |
| Decision Tree | 0.623 | 0.459 | 0.455 | 0.457 | 0.621 |
| Random Forest | 0.639 | 0.497 | 0.514 | 0.503 | 0.644 |
| KNN | 0.678 | 0.510 | 0.494 | 0.500 | 0.671 |
| Gaussian Naive Bayes | 0.442 | 0.538 | 0.565 | 0.411 | 0.530 |

**Recall on the High class:**
- Gaussian NB 0.85 and Logistic Regression 0.60, but both with many false "High" predictions.
- Random Forest 0.25, Decision Tree 0.11 and KNN 0.11.

### Best model: Logistic Regression

- **Why macro F1.** It is the selection metric because it treats Low, Medium and High as equally important. Accuracy would favour models that ignore the rare High class.
- **Result.** Logistic Regression has the highest macro F1 (0.507). Random Forest (0.503) and KNN (0.500) are very close.
- **Its strength.** It finds 60 % of the High tracks, against 25 % for Random Forest and 11 % for KNN.
- **Trade-off.** Its High-class precision is low (0.13), so many tracks it labels High are really Low or Medium. KNN has the highest accuracy (0.678) but rarely finds High tracks.
- **Saved model.** The model, scaler, column list and fill values are saved together in `models/best_model.joblib`. The file was reloaded and gave identical predictions.

**Most useful inputs.** Genre accounts for about 25 % of the Random Forest importance. The audio features contribute 5–8 % each, and explicit, mode and time signature contribute very little.

**Conclusion.** Genre and audio features carry *some* information about popularity: macro F1 is about 0.5, against 0.24 for always guessing "Low". Most of popularity, however, depends on things not in the data, such as artist fame, marketing and release date.

## Key findings, conclusions and recommendations

These are also printed at the end of `step5_train_models.py`.

**Key findings**
1. Genre is the strongest single factor: it explains about 26 % of the variation in popularity.
2. No audio feature is more than weakly related to popularity (all |r| about 0.10 or less).
3. Audio features differ strongly *between genres* (eta² 0.31–0.50 for six of seven features).
4. Explicit tracks are only slightly more popular (d = 0.16); their real difference is far more speech (d = 1.28).
5. The best classifier (Logistic Regression) reaches a macro F1 of 0.507. Genre is the most useful input.

**Overall conclusion.** Genre and audio features carry some information about popularity, but most of what makes a track popular isn't captured in this dataset.

**Recommendations**
- **For artists and labels:** popularity depends far more on genre and outside factors than on sound. Changing danceability or energy alone is unlikely to create a hit.
- **For streaming platforms:** audio features are strongly informative about genre. Use them for genre-based recommendations and playlists rather than for predicting popularity.
- **Choosing a model:** use Logistic Regression with balanced weights when finding potential hits matters most (High recall 0.60). Use KNN when overall accuracy matters most (0.678).
- **For better predictions:** add artist follower counts, release date, playlist placements and marketing data.
- **On data quality:** investigate the many zero-popularity tracks in genres such as iranian, jazz, romance and latin before relying on those genres' results.

## Coursework requirements checklist

| Requirement | Where it is done |
|---|---|
| Data preprocessing and cleaning: missing values, duplicates, inconsistent data | `step1_main.py` (detection), `step2_preprocessing.py` (cleaning). Covers the missing-name record, missing tempo/time signature, exact duplicates, repeated tracks, conflicting popularity for the same track, extra spaces and genre spelling |
| Exploratory Data Analysis | `step3_eda.py`: 6 charts plus statistics and interpretations |
| At least five objectives | `step4_objectives.py` (objectives 1–4) and `step5_train_models.py` (objective 5) |
| Data visualisation | 18 charts: histograms, box plots, scatter plots, a heatmap, bar charts and confusion matrices |
| Feature engineering | `step2_preprocessing.py` (target, `duration_min`, `num_artists`, `rhythm_undetected`) and `step5_train_models.py` (imputation, one-hot encoding, scaling) |
| At least 4 ML models and the best-fit model | 5 models in `step5_train_models.py`. Best model chosen by macro F1 |
| Evaluation with appropriate metrics | Accuracy, precision, recall, macro and weighted F1, confusion matrices and classification reports (per-class precision and recall) |
| Key findings, conclusions and recommendations | End of `step5_train_models.py` and the section above |

## Saved figures (`outputs/figures/`)

| EDA | Objectives | Models |
|---|---|---|
| `01_popularity_distribution.png` | `07_obj1_genre_popularity.png` | `12_cm_logistic_regression.png` |
| `02_duration_distribution.png` | `08_obj2_feature_correlations.png` | `13_cm_decision_tree.png` |
| `03_features_vs_popularity.png` | `09_obj3_genre_effect_sizes.png` | `14_cm_random_forest.png` |
| `04_correlation_heatmap.png` | `10_obj3_top_feature_by_genre.png` | `15_cm_knn.png` |
| `05_feature_boxplots.png` | `11_obj4_explicit_comparison.png` | `16_cm_gaussian_nb.png` |
| `06_popularity_category_balance.png` | | `17_model_comparison.png` |
| | | `18_feature_importance.png` |

The correlation heatmap shows only the lower-left triangle on purpose. The matrix is symmetric (energy–loudness equals loudness–energy), so the upper-right half would repeat the same numbers.
## Limitations

- The analysis uses a 10,000-row stratified sample, not all 114,000 rows. Small genre-level numbers can vary with a different sample.
- Popularity is a snapshot that depends on when the data was collected. Zero-popularity spikes in some genres (iranian, romance, latin, jazz) may be collection artefacts.
- The Low/Medium/High thresholds are arbitrary, and the High class is small (89 test tracks).
- The data is observational, so no causal claims can be made.
- Important drivers of popularity (artist following, release date, playlists, marketing) are not in the dataset.
- Tracks listed under several genres were kept under their first genre only.

## Future improvements

- Repeat the analysis on the full dataset, or on several random samples, to check stability.
- Add release year and artist-level information from the Spotify API.
- Try different class thresholds or predict the numeric score with regression.
- Use probability thresholds to balance precision and recall for the High class.

---

## Screenshot checklist

The `screenshots/` folders are empty and ready. No screenshots have been created or faked. Please capture these yourself while running the project in VS Code (Windows: **Win + Shift + S**):

**`screenshots/vscode/`**
1. Project open in VS Code, with the Explorer showing the folder structure.
2. `step1_main.py` terminal output: shape, columns and types.
3. `step1_main.py` output: missing values and duplicates sections.
4. `step2_preprocessing.py` output: sampling table and cleaning steps with row counts.
5. `step3_eda.py` terminal output: statistics and interpretations.
6. `step4_objectives.py` terminal output for each objective (1–4).
7. `step5_train_models.py` output: each model's results and the model comparison table.
8. `step5_train_models.py` output: best model, feature importance, "loaded model gives the same predictions: True" and the recommendations.
9. The Explorer showing the final outputs (`outputs/`, `models/`, `data/processed/`).

**`screenshots/visualizations/`**: a screenshot of each chart window as it appears on screen. The PNG files in `outputs/figures/` can also be used directly.

**`screenshots/linkedin/`** and **`screenshots/github/`**: placeholders. Add your own genuine screenshots later.

## GitHub

This folder is ready to publish (`.gitignore` included). The raw dataset is about 20 MB, which is under GitHub's 100 MB file limit. **Nothing has been uploaded to GitHub.**
