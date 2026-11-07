import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from sentiment import ROOT


class AppTests(unittest.TestCase):
    def make_app(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        self.assertEqual(len(app.exception), 0)
        return app

    def submit_single(self, app):
        next(button for button in app.button if button.label == "Analyze sentiment").click().run()
        self.assertEqual(len(app.exception), 0)

    def test_startup_does_not_train(self):
        with patch("sklearn.pipeline.Pipeline.fit", side_effect=AssertionError("Startup must not train")):
            app = self.make_app()
        self.assertEqual(len(app.tabs), 3)
        self.assertTrue(any(metric.label == "Holdout accuracy" for metric in app.metric))

    def test_single_prediction_and_empty_feedback(self):
        app = self.make_app()
        app.selectbox(key="example_choice").select("Positive example").run()
        self.submit_single(app)
        self.assertTrue(any('result-name positive' in element.value for element in app.markdown))
        app.text_area(key="feedback_input").set_value("")
        self.submit_single(app)
        self.assertTrue(any("Empty feedback" in warning.value for warning in app.warning))

    def test_sample_batch_and_filter(self):
        app = self.make_app()
        app.button(key="analyze_batch").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.session_state["batch_result"]), 12)
        app.selectbox(key="sentiment_filter").select("Negative").run()
        result_tables = [table.value for table in app.dataframe if "status" in table.value.columns]
        self.assertEqual(len(result_tables), 1)
        self.assertTrue(result_tables[0].sentiment.eq("negative").all())
        self.assertGreater(len(result_tables[0]), 0)


if __name__ == "__main__":
    unittest.main()
