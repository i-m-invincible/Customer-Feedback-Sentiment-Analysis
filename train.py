"""Train and evaluate the same TF-IDF + SVM pipeline served by the app."""

import argparse
import gzip
import hashlib
import json
import pickle
import platform
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from sentiment import LABELS, ROOT, clean_text


def train(data_path: Path, output_dir: Path) -> dict:
    source_bytes = data_path.read_bytes()
    data = pd.read_csv(data_path, keep_default_na=False)
    if not {"text", "sentiment"}.issubset(data.columns):
        raise ValueError("Training data needs text and sentiment columns.")
    data["text"] = data["text"].map(clean_text)
    data["sentiment"] = data["sentiment"].str.lower().str.strip()
    valid = data["text"].str.contains(r"[a-zA-Z]", regex=True)
    data = data[valid & data["sentiment"].isin(LABELS)].copy()
    # Remove contradictory labels and repeated text before splitting.
    conflicts = data.groupby("text")["sentiment"].transform("nunique") > 1
    data = data[~conflicts].drop_duplicates("text").reset_index(drop=True)
    if set(data["sentiment"]) != set(LABELS):
        raise ValueError("Training data must contain negative, neutral, and positive feedback.")
    x_train, x_test, y_train, y_test = train_test_split(
        data["text"], data["sentiment"], test_size=0.20,
        random_state=42, stratify=data["sentiment"],
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=clean_text, lowercase=False, strip_accents="unicode",
            ngram_range=(1, 2), min_df=2, max_df=0.98,
            max_features=30_000, sublinear_tf=True,
        )),
        ("svm", LinearSVC(
            C=0.7, class_weight="balanced", dual="auto",
            random_state=42, max_iter=6_000,
        )),
    ])
    # Fit the vectorizer on training text only. Keep the test set untouched.
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    majority_label = y_train.value_counts().idxmax()
    payload = gzip.compress(pickle.dumps(model, protocol=5), mtime=0)
    metadata = {
        "model": "TF-IDF (word unigrams and bigrams) + LinearSVC",
        "dataset": "CrowdFlower / Twitter US Airline Sentiment (February 2015)",
        "dataset_url": "https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment",
        "dataset_license": "CC BY-NC-SA 4.0 (see data/SOURCES.md)",
        "source_rows": len(pd.read_csv(data_path)),
        "usable_rows": len(data),
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "random_state": 42,
        "split": "80/20 stratified holdout after normalized-text deduplication",
        "class_counts": {key: int(value) for key, value in data["sentiment"].value_counts().items()},
        "accuracy": float(accuracy_score(y_test, predicted)),
        "macro_f1": float(f1_score(y_test, predicted, average="macro")),
        "majority_baseline_accuracy": float((y_test == majority_label).mean()),
        "classification_report": classification_report(
            y_test, predicted, labels=list(LABELS), output_dict=True, zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_test, predicted, labels=list(LABELS)).tolist(),
        "labels": list(LABELS),
        "vocabulary_size": len(model.named_steps["tfidf"].vocabulary_),
        "data_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "model_sha256": hashlib.sha256(payload).hexdigest(),
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
        },
        "limitations": [
            "English airline feedback; accuracy may differ in other business domains.",
            "Sarcasm, mixed sentiment, short messages, and unseen wording can be misclassified.",
            "No probability or confidence percentage is produced by this LinearSVC model.",
            "The bundled model is the exact training-split model evaluated here; it is not refitted on test data.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sentiment_model.pkl.gz").write_bytes(payload)
    (output_dir / "metrics.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "feedback.csv.gz")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()
    result = train(args.data, args.output)
    print(f"Accuracy: {result['accuracy']:.2%} | Macro F1: {result['macro_f1']:.4f}")
    print(f"Training rows: {result['train_rows']:,} | Holdout rows: {result['test_rows']:,}")
