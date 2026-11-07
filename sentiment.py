"""Text preparation, trusted model loading, and feedback analysis."""

import csv
import gzip
import hashlib
import html
import io
import json
import pickle
import re
import unicodedata
from pathlib import Path

import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parent
LABELS = ("negative", "neutral", "positive")
MAX_ROWS = 5_000
MAX_CHARACTERS = 5_000
MAX_FILE_BYTES = 5 * 1024 * 1024


def clean_text(text: str) -> str:
    """Keep words and negation; remove handles, links, and extra spaces."""
    text = unicodedata.normalize("NFKC", html.unescape(str(text))).lower()
    text = text.replace("’", "'").replace("‘", "'")
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"(?<!\w)@\w+", " ", text)
    text = re.sub(r"\brt\b(?=\s|:)", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_model(artifact_dir: Path | None = None):
    """Load only the repository's bundled, checksum-verified pickle."""
    artifact_dir = Path(artifact_dir) if artifact_dir else ROOT / "artifacts"
    metadata = json.loads((artifact_dir / "metrics.json").read_text(encoding="utf-8"))
    payload = (artifact_dir / "sentiment_model.pkl.gz").read_bytes()
    if hashlib.sha256(payload).hexdigest() != metadata["model_sha256"]:
        raise ValueError("The bundled model checksum does not match its metadata.")
    if sklearn.__version__ != metadata["versions"]["scikit-learn"]:
        raise ValueError("Install requirements.txt to use the matching scikit-learn version.")
    # This path is fixed, never supplied by an app visitor. Never load user pickles.
    model = pickle.loads(gzip.decompress(payload))
    if tuple(model.classes_) != LABELS:
        raise ValueError("The bundled model must support all three sentiment classes.")
    return model, metadata


def analyze_feedback(model, values) -> pd.DataFrame:
    """Preserve every input row, including rows that cannot be classified."""
    values = list(values)
    if len(values) > MAX_ROWS:
        raise ValueError(f"Use at most {MAX_ROWS:,} feedback entries at a time.")
    rows = []
    positions = []
    cleaned = []
    for position, value in enumerate(values):
        text = value if isinstance(value, str) else ""
        status = "Analyzed"
        if not isinstance(value, str):
            status = "Missing or non-text feedback"
        elif len(text) > MAX_CHARACTERS:
            status = "Feedback exceeds 5,000 characters"
        elif not clean_text(text):
            status = "Empty feedback"
        rows.append({"feedback": text, "sentiment": "unclassified", "status": status})
        if status == "Analyzed":
            positions.append(position)
            cleaned.append(text)
    if cleaned:
        features = model.named_steps["tfidf"].transform(cleaned)
        known = features.getnnz(axis=1) > 0
        predictions = model.named_steps["svm"].predict(features)
        for row, prediction, has_words in zip(positions, predictions, known):
            if has_words:
                rows[row]["sentiment"] = str(prediction)
            else:
                rows[row]["status"] = "No known vocabulary"
    return pd.DataFrame(rows, columns=["feedback", "sentiment", "status"])


def read_feedback_csv(payload: bytes) -> pd.DataFrame:
    if len(payload) > MAX_FILE_BYTES:
        raise ValueError("Choose a CSV smaller than 5 MB.")
    try:
        frame = pd.read_csv(
            io.BytesIO(payload), encoding="utf-8-sig", dtype=str,
            keep_default_na=False, nrows=MAX_ROWS + 1,
        )
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise ValueError("Upload a valid UTF-8 CSV with a header row.") from exc
    if frame.empty:
        raise ValueError("The CSV needs at least one feedback row.")
    if len(frame) > MAX_ROWS:
        raise ValueError(f"Use at most {MAX_ROWS:,} feedback rows per CSV.")
    return frame


def export_csv(frame: pd.DataFrame) -> bytes:
    """Protect spreadsheet downloads against formula-like customer text."""
    def safe_cell(value):
        if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
            return "'" + value
        if isinstance(value, str) and value.startswith(("\t", "\r", "\n")):
            return "'" + value
        return value

    safe = frame.map(safe_cell)
    return safe.to_csv(index=False, quoting=csv.QUOTE_ALL).encode("utf-8-sig")
