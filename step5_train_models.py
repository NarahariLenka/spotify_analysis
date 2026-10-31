"""
STEP 5 - Objective 5: Machine-learning popularity classification
================================================================
Run fifth:   python step5_train_models.py      (needs: python step2_preprocessing.py)

Question: can we predict whether a track's popularity is Low (0-39),
Medium (40-69) or High (70-100) from its genre and audio features?

Five models are trained one after another on the SAME training data and
tested on the SAME test data:
  1. Logistic Regression   2. Decision Tree   3. Random Forest
  4. KNN                   5. Gaussian Naive Bayes
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

PROJECT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = PROJECT_DIR / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
(PROJECT_DIR / "models").mkdir(exist_ok=True)

CLASSES = ["Low", "Medium", "High"]

# ---------------------------------------------------------------------------
# Load the cleaned data
# ---------------------------------------------------------------------------
df = pd.read_csv(PROJECT_DIR / "data" / "processed" / "spotify_tracks_cleaned.csv")
print(df.info())
print(df["popularity_category"].value_counts())

# ---------------------------------------------------------------------------
# Choose inputs (X) and target (y)
# ---------------------------------------------------------------------------
# NOT used as inputs:
#   popularity, popularity_category -> this is the answer we want to predict
#   track_id                         -> just an ID number
#   artists, album_name, track_name  -> thousands of different names
feature_columns = ["duration_min", "danceability", "energy", "loudness", "speechiness",
                   "acousticness", "instrumentalness", "liveness", "valence", "tempo",
                   "num_artists", "explicit", "mode", "rhythm_undetected",
                   "track_genre", "key", "time_signature"]
X = df[feature_columns]
y = df["popularity_category"]

# ---------------------------------------------------------------------------
# Split into training (80%) and test (20%) data
# ---------------------------------------------------------------------------
# stratify=y keeps the same share of Low/Medium/High in both parts.
# The split is done FIRST, so nothing about the test data is used for training.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print("Training rows:", len(X_train), " Test rows:", len(X_test))

# ---------------------------------------------------------------------------
# Prepare the data (values learned from the TRAINING data only)
# ---------------------------------------------------------------------------
# 1. Fill missing values (tempo / time_signature were 0 = "not detected")
tempo_median = X_train["tempo"].median()
time_signature_mode = X_train["time_signature"].mode()[0]
X_train = X_train.fillna({"tempo": tempo_median, "time_signature": time_signature_mode})
X_test = X_test.fillna({"tempo": tempo_median, "time_signature": time_signature_mode})

# 2. One-hot encode the categories (genre, key, time signature become 0/1 columns)
X_train = pd.get_dummies(X_train, columns=["track_genre", "key", "time_signature"])
X_test = pd.get_dummies(X_test, columns=["track_genre", "key", "time_signature"])
# give the test data exactly the same columns as the training data
X_train_columns = X_train.columns
X_test = X_test.reindex(columns=X_train_columns, fill_value=0)

# 3. Scale every column to mean 0 and standard deviation 1 (needed by KNN and
#    Logistic Regression). The scaler is fitted on the training data only.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Number of input columns after encoding:", X_train.shape[1])

# The results of every model are collected here for the comparison at the end
results = []

# ---------------------------------------------------------------------------
# 1. Logistic Regression
# ---------------------------------------------------------------------------
# class_weight="balanced" gives the rare High class more importance
print("\n========== Logistic Regression ==========")
model = LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred, labels=CLASSES))
print("Classification Report:\n", classification_report(y_test, y_pred))
results.append(["Logistic Regression",
                accuracy_score(y_test, y_pred),
                precision_score(y_test, y_pred, average="macro"),
                recall_score(y_test, y_pred, average="macro"),
                f1_score(y_test, y_pred, average="macro"),
                f1_score(y_test, y_pred, average="weighted")])
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, labels=CLASSES, cmap="Blues")
plt.title("Confusion matrix - Logistic Regression")
plt.savefig(FIGURES_DIR / "12_cm_logistic_regression.png", dpi=200, bbox_inches="tight")
plt.show()
logistic_model = model

# ---------------------------------------------------------------------------
# 2. Decision Tree
# ---------------------------------------------------------------------------
print("\n========== Decision Tree ==========")
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred, labels=CLASSES))
print("Classification Report:\n", classification_report(y_test, y_pred))
results.append(["Decision Tree",
                accuracy_score(y_test, y_pred),
                precision_score(y_test, y_pred, average="macro"),
                recall_score(y_test, y_pred, average="macro"),
                f1_score(y_test, y_pred, average="macro"),
                f1_score(y_test, y_pred, average="weighted")])
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, labels=CLASSES, cmap="Blues")
plt.title("Confusion matrix - Decision Tree")
plt.savefig(FIGURES_DIR / "13_cm_decision_tree.png", dpi=200, bbox_inches="tight")
plt.show()
tree_model = model

# ---------------------------------------------------------------------------
# 3. Random Forest
# ---------------------------------------------------------------------------
print("\n========== Random Forest ==========")
model = RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                               class_weight="balanced", random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred, labels=CLASSES))
print("Classification Report:\n", classification_report(y_test, y_pred))
results.append(["Random Forest",
                accuracy_score(y_test, y_pred),
                precision_score(y_test, y_pred, average="macro"),
                recall_score(y_test, y_pred, average="macro"),
                f1_score(y_test, y_pred, average="macro"),
                f1_score(y_test, y_pred, average="weighted")])
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, labels=CLASSES, cmap="Blues")
plt.title("Confusion matrix - Random Forest")
plt.savefig(FIGURES_DIR / "14_cm_random_forest.png", dpi=200, bbox_inches="tight")
plt.show()
forest_model = model

# ---------------------------------------------------------------------------
# 4. K-Nearest Neighbors (KNN)
# ---------------------------------------------------------------------------
print("\n========== KNN ==========")
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred, labels=CLASSES))
print("Classification Report:\n", classification_report(y_test, y_pred))
results.append(["KNN",
                accuracy_score(y_test, y_pred),
                precision_score(y_test, y_pred, average="macro"),
                recall_score(y_test, y_pred, average="macro"),
                f1_score(y_test, y_pred, average="macro"),
                f1_score(y_test, y_pred, average="weighted")])
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, labels=CLASSES, cmap="Blues")
plt.title("Confusion matrix - KNN")
plt.savefig(FIGURES_DIR / "15_cm_knn.png", dpi=200, bbox_inches="tight")
plt.show()
knn_model = model

# ---------------------------------------------------------------------------
# 5. Gaussian Naive Bayes
# ---------------------------------------------------------------------------
# var_smoothing adds a little extra variance; this keeps the many 0/1 genre
# columns from giving extreme probabilities.
print("\n========== Gaussian Naive Bayes ==========")
model = GaussianNB(var_smoothing=1e-3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred, labels=CLASSES))
print("Classification Report:\n", classification_report(y_test, y_pred))
results.append(["Gaussian Naive Bayes",
                accuracy_score(y_test, y_pred),
                precision_score(y_test, y_pred, average="macro"),
                recall_score(y_test, y_pred, average="macro"),
                f1_score(y_test, y_pred, average="macro"),
                f1_score(y_test, y_pred, average="weighted")])
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, labels=CLASSES, cmap="Blues")
plt.title("Confusion matrix - Gaussian Naive Bayes")
plt.savefig(FIGURES_DIR / "16_cm_gaussian_nb.png", dpi=200, bbox_inches="tight")
plt.show()
naive_bayes_model = model

# ---------------------------------------------------------------------------
# Compare the five models
# ---------------------------------------------------------------------------
comparison = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision (macro)",
                                            "Recall (macro)", "F1 (macro)", "F1 (weighted)"])
comparison = comparison.set_index("Model").round(4)
print("\n========== MODEL COMPARISON ==========")
print(comparison.to_string())
comparison.to_csv(PROJECT_DIR / "outputs" / "model_comparison.csv")

comparison.plot(kind="bar", figsize=(12, 6))
plt.title("Comparison of the five models on the test data")
plt.xlabel("Model")
plt.ylabel("Score (0-1)")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.savefig(FIGURES_DIR / "17_model_comparison.png", dpi=200, bbox_inches="tight")
plt.show()

# ---------------------------------------------------------------------------
# Best model
# ---------------------------------------------------------------------------
# Macro F1 is used because it treats Low, Medium and High as equally
# important. Accuracy would favour a model that ignores the rare High class.
best_name = comparison["F1 (macro)"].idxmax()
print("\nBEST MODEL (highest macro F1):", best_name)
print(comparison.loc[best_name])

all_models = {"Logistic Regression": logistic_model, "Decision Tree": tree_model,
              "Random Forest": forest_model, "KNN": knn_model,
              "Gaussian Naive Bayes": naive_bayes_model}
best_model = all_models[best_name]

# ---------------------------------------------------------------------------
# Which inputs are most useful? (Random Forest feature importance)
# ---------------------------------------------------------------------------
importance = pd.Series(forest_model.feature_importances_, index=X_train_columns)
# add the 0/1 columns of each category back together, e.g. all "track_genre_..." columns
for category in ["track_genre", "key", "time_signature"]:
    category_columns = [c for c in importance.index if c.startswith(category + "_")]
    importance[category] = importance[category_columns].sum()
    importance = importance.drop(category_columns)
importance = importance.sort_values(ascending=False)
print("\nRandom Forest feature importance:")
print(importance.round(4))

importance.plot(kind="barh", figsize=(9, 6), color="#55a868")
plt.gca().invert_yaxis()
plt.title("Which inputs are most useful? (Random Forest feature importance)")
plt.xlabel("Importance (all bars add up to 1)")
plt.ylabel("Input feature")
plt.savefig(FIGURES_DIR / "18_feature_importance.png", dpi=200, bbox_inches="tight")
plt.show()

# ---------------------------------------------------------------------------
# Save the best model and check that it works after loading
# ---------------------------------------------------------------------------
# The scaler, column names and fill values are saved too, because new data
# must be prepared in exactly the same way before predicting.
saved = {"model": best_model, "scaler": scaler, "columns": list(X_train_columns),
         "tempo_median": tempo_median, "time_signature_mode": time_signature_mode}
joblib.dump(saved, PROJECT_DIR / "models" / "best_model.joblib")

loaded = joblib.load(PROJECT_DIR / "models" / "best_model.joblib")
check = (loaded["model"].predict(X_test) == best_model.predict(X_test)).all()
print("\nSaved models/best_model.joblib - loaded model gives the same predictions:", check)

# ---------------------------------------------------------------------------
# Key findings, conclusions and recommendations (whole project)
# ---------------------------------------------------------------------------
print("\n========== KEY FINDINGS ==========")
print("1. Popularity differs strongly between genres (genre explains about a quarter of it).")
print("2. No single audio feature is more than weakly related to popularity (all |r| about 0.10 or less).")
print("3. Audio features differ strongly between genres - they describe a genre's sound.")
print("4. Explicit tracks are only slightly more popular; mainly they contain more speech.")
print(f"5. Best model: {best_name} (macro F1 = {comparison.loc[best_name, 'F1 (macro)']:.3f}); "
      "genre is the most useful input.")
print("\nCONCLUSION: genre and audio features give SOME information about popularity, but most "
      "of what makes a track popular (artist fame, marketing, release date) is not in this dataset.")
print("\n========== RECOMMENDATIONS ==========")
print("- Artists / labels: changing danceability or energy alone is unlikely to create a hit.")
print("- Streaming platforms: use audio features for genre-based playlists and recommendations.")
most_accurate = comparison["Accuracy"].idxmax()
print(f"- Use {best_name} when finding tracks of every class (including High) matters most,")
print(f"  and {most_accurate} (highest accuracy) when overall accuracy matters most.")
print("- Add artist followers, release date and playlist data to improve predictions.")
print("- Check the many zero-popularity tracks in some genres before relying on them.")
print("\nAll steps finished.")
