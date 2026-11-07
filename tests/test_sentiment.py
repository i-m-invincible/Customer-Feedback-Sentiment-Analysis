import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from sentiment import (
    LABELS, MAX_ROWS, ROOT, analyze_feedback, clean_text,
    export_csv, load_model, read_feedback_csv,
)


class SentimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.metrics = load_model()

    def test_positive_neutral_and_negative_examples(self):
        examples = [
            "Thank you for the excellent service. The staff were friendly and helpful.",
            "What time does the flight leave tomorrow?",
            "My flight was cancelled and nobody helped me get a refund.",
        ]
        results = analyze_feedback(self.model, examples)
        self.assertEqual(results.sentiment.tolist(), ["positive", "neutral", "negative"])

    def test_empty_unknown_and_overlong_text_remain_unclassified(self):
        results = analyze_feedback(self.model, ["  ", None, "https://example.com", "zzzxxyyqq", "a" * 5001])
        self.assertEqual(len(results), 5)
        self.assertTrue(results.sentiment.eq("unclassified").all())
        self.assertTrue(results.status.ne("Analyzed").all())

    def test_mixed_rows_preserve_text_and_order(self):
        inputs = ["Thanks for the amazing service!", "", "Where is my flight?"]
        results = analyze_feedback(self.model, inputs)
        self.assertEqual(results.feedback.tolist(), inputs)
        self.assertEqual(results.iloc[1].status, "Empty feedback")
        self.assertEqual(len(results), len(inputs))

    def test_cleanup_keeps_negation(self):
        self.assertEqual(clean_text("@Airline NOT happy &amp; not helpful https://example.com"), "not happy & not helpful")

    def test_csv_preserves_literal_na_and_accepts_bom(self):
        frame = read_feedback_csv(b'\xef\xbb\xbffeedback,other\n"NA",1\n"Thanks, great service!",2\n')
        self.assertEqual(frame.feedback.tolist(), ["NA", "Thanks, great service!"])

    def test_invalid_csv_and_limits(self):
        for payload in (b"", b"feedback\n", b"feedback\n\xff", b'feedback\n"unclosed'):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                read_feedback_csv(payload)
        with self.assertRaises(ValueError):
            read_feedback_csv(b"a" * (5 * 1024 * 1024 + 1))
        with self.assertRaises(ValueError):
            read_feedback_csv(("feedback\n" + "hello\n" * (MAX_ROWS + 1)).encode())
        with self.assertRaises(ValueError):
            analyze_feedback(self.model, [""] * (MAX_ROWS + 1))

    def test_export_neutralizes_formulas(self):
        inputs = ["=1+1", "  +SUM(1,1)", "@someone", "\tformula", "ordinary feedback"]
        exported = pd.read_csv(io.BytesIO(export_csv(pd.DataFrame({"feedback": inputs}))))
        self.assertTrue(all(text.startswith("'") for text in exported.feedback.iloc[:4]))
        self.assertEqual(exported.feedback.iloc[4], inputs[4])

    def test_altered_model_is_rejected_before_unpickling(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "metrics.json").write_text(json.dumps(self.metrics), encoding="utf-8")
            (path / "sentiment_model.pkl.gz").write_bytes(b"changed model")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_model(path)

    def test_published_metrics_match_bundled_model_on_disjoint_holdout(self):
        source = ROOT / "data" / "feedback.csv.gz"
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), self.metrics["data_sha256"])
        data = pd.read_csv(source, keep_default_na=False)
        data["text"] = data.text.map(clean_text)
        data = data[data.text.str.contains(r"[a-zA-Z]", regex=True) & data.sentiment.isin(LABELS)].copy()
        conflict = data.groupby("text").sentiment.transform("nunique") > 1
        data = data[~conflict].drop_duplicates("text").reset_index(drop=True)
        x_train, x_test, _, y_test = train_test_split(data.text, data.sentiment, test_size=.2, random_state=42, stratify=data.sentiment)
        self.assertTrue(set(x_train).isdisjoint(set(x_test)))
        predicted = self.model.predict(x_test)
        self.assertAlmostEqual(accuracy_score(y_test, predicted), self.metrics["accuracy"])
        self.assertAlmostEqual(f1_score(y_test, predicted, average="macro"), self.metrics["macro_f1"])


if __name__ == "__main__":
    unittest.main()
