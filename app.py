"""Streamlit interface for customer feedback sentiment analysis."""

import hashlib

import pandas as pd
import streamlit as st

from sentiment import (
    LABELS, MAX_CHARACTERS, ROOT, analyze_feedback, analyze_feedback_frame, export_csv,
    load_model, read_feedback_csv,
)

st.set_page_config(
    page_title="Customer Feedback | Sentiment Analysis",
    page_icon="◍", layout="wide", initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .block-container { max-width: 1160px; padding-top: 3rem; padding-bottom: 3rem; }
    h1 { font-size: clamp(2.1rem, 4vw, 3.25rem) !important; letter-spacing: -.055em;
         font-weight: 650 !important; line-height: 1.08 !important; }
    h2, h3 { letter-spacing: -.025em; }
    .eyebrow { font-size: .72rem; letter-spacing: .18em; color: #66766C;
                font-weight: 650; margin: 0 0 1rem; }
    .intro { color: #657168; font-size: 1.04rem; max-width: 640px; line-height: 1.65; }
    .brand { font-size: .85rem; font-weight: 650; letter-spacing: -.02em; }
    .status { font-size: .76rem; color: #17624B; text-align: right; padding-top: .1rem; }
    [data-testid="stTabs"] { margin-top: 1rem; }
    [data-baseweb="tab-list"] { gap: 2rem; border-bottom: 1px solid #DBE0D7; }
    [data-baseweb="tab"] { padding: .8rem 0; }
    [data-testid="stMetricValue"] { letter-spacing: -.04em; }
    [data-testid="stVerticalBlockBorderWrapper"] > div { border-radius: 12px; }
    .result-label { text-transform: uppercase; letter-spacing: .14em;
                    font-size: .68rem; color: #6B776E; margin-bottom: .8rem; }
    .result-name { font-size: 2.5rem; letter-spacing: -.05em; font-weight: 600;
                   line-height: 1.15; margin-bottom: .7rem; }
    .positive { color: #17624B; } .negative { color: #A34D38; } .neutral { color: #626D76; }
    .empty-result { color: #869087; font-size: 1.2rem; line-height: 1.5; }
    .footer { border-top: 1px solid #DBE0D7; margin-top: 2rem; padding-top: 1rem;
              color: #7A857D; font-size: .74rem; }
    @media (max-width: 640px) {
       .block-container { padding-top: 1.5rem; }
       [data-baseweb="tab-list"] { gap: 1rem; }
       .status { text-align: left; }
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading the sentiment model…")
def get_model():
    return load_model()


try:
    model, metrics = get_model()
except (OSError, ValueError, KeyError) as error:
    st.error("The sentiment model could not be loaded.")
    st.caption(str(error))
    st.code("python -m pip install -r requirements.txt\npython train.py")
    st.stop()

brand, status = st.columns([3, 1])
brand.markdown('<div class="brand">◍ &nbsp; CUSTOMER FEEDBACK</div>', unsafe_allow_html=True)
status.markdown('<div class="status">● &nbsp; Sentiment model ready</div>', unsafe_allow_html=True)
st.write("")
st.markdown('<div class="eyebrow">CUSTOMER EXPERIENCE / SENTIMENT ANALYSIS</div>', unsafe_allow_html=True)
st.title("Customer Feedback\nSentiment Analysis")
st.markdown(
    '<p class="intro">Find the positive, neutral, and negative in customer feedback. '
    'Read one response or explore the bigger picture in a CSV.</p>', unsafe_allow_html=True,
)

single_tab, batch_tab, model_tab = st.tabs(["Single feedback", "Feedback overview", "Model & results"])

EXAMPLES = {
    "Write your own": "",
    "Positive example": "Thank you for the excellent service. The staff were friendly and helpful.",
    "Neutral example": "What time does the flight leave tomorrow?",
    "Negative example": "My flight was cancelled and nobody helped me get a refund.",
}


def choose_example():
    st.session_state["feedback_input"] = EXAMPLES[st.session_state["example_choice"]]
    st.session_state.pop("single_result", None)


with single_tab:
    st.write("")
    entry, result_panel = st.columns([1.45, 1], gap="large")
    with entry:
        st.subheader("What did your customer say?")
        st.selectbox("Try an example", EXAMPLES, key="example_choice", on_change=choose_example)
        with st.form("single_feedback"):
            feedback = st.text_area(
                "Customer feedback", key="feedback_input", height=175,
                max_chars=MAX_CHARACTERS,
                placeholder="Paste a customer review, message, or support response…",
            )
            submitted = st.form_submit_button("Analyze sentiment", type="primary")
        if submitted:
            st.session_state["single_result"] = analyze_feedback(model, [feedback]).iloc[0].to_dict()
        st.caption("English feedback · Up to 5,000 characters")
    with result_panel:
        st.write("")
        with st.container(border=True):
            st.markdown('<div class="result-label">SENTIMENT RESULT</div>', unsafe_allow_html=True)
            result = st.session_state.get("single_result")
            if result and result["feedback"] == feedback:
                label = result["sentiment"]
                if label in LABELS:
                    st.markdown(f'<div class="result-name {label}">{label.title()}</div>', unsafe_allow_html=True)
                    descriptions = {
                        "positive": "The feedback expresses satisfaction or appreciation.",
                        "neutral": "The feedback is informational or has no clear positive or negative tone.",
                        "negative": "The feedback expresses dissatisfaction or a concern worth reviewing.",
                    }
                    st.write(descriptions[label])
                    st.caption("Model prediction · Check the original message before acting.")
                else:
                    st.warning(result["status"] + ". Try a complete sentence in English.")
            else:
                st.markdown('<div class="empty-result">Your result will appear here.</div>', unsafe_allow_html=True)
                st.caption("Add feedback and select Analyze sentiment to see its classification.")
        st.caption("Trained on airline customer feedback. Other business domains may perform differently.")


with batch_tab:
    st.write("")
    st.subheader("See the bigger picture")
    st.caption("Analyze a collection of responses, then review and download the results.")
    source = st.radio("Feedback source", ["Sample feedback", "Upload CSV"], horizontal=True, key="batch_source")
    frame = None
    signature = None
    column = None
    if source == "Sample feedback":
        payload = (ROOT / "data" / "sample_feedback.csv").read_bytes()
        frame = read_feedback_csv(payload)
        column = "feedback"
        signature = ("sample", hashlib.sha256(payload).hexdigest(), column)
        st.caption("12 illustrative examples, separate from the training and evaluation data.")
    else:
        uploaded = st.file_uploader("Upload customer feedback", type=["csv"], key="feedback_csv")
        st.caption("UTF-8 CSV · Header row required · Up to 5 MB and 5,000 rows")
        if uploaded is not None:
            try:
                payload = uploaded.getvalue()
                frame = read_feedback_csv(payload)
                default_index = next((i for i, name in enumerate(frame.columns) if name.lower() in {"feedback", "text", "review"}), 0)
                column = st.selectbox("Feedback column", frame.columns, index=default_index, key="feedback_column")
                signature = ("upload", hashlib.sha256(payload).hexdigest(), column)
            except ValueError as error:
                st.error(str(error))
    if frame is not None and column is not None:
        with st.expander(f"Preview {len(frame):,} feedback entries"):
            st.dataframe(frame[[column]].head(10), hide_index=True, width="stretch")
        if st.button("Analyze feedback", type="primary", key="analyze_batch"):
            with st.spinner("Analyzing customer feedback…"):
                st.session_state["batch_result"] = analyze_feedback_frame(model, frame, column)
            st.session_state["batch_signature"] = signature
        if st.session_state.get("batch_signature") == signature:
            results = st.session_state["batch_result"]
            classified = results[results["sentiment"].isin(LABELS)]
            counts = classified["sentiment"].value_counts().reindex(LABELS, fill_value=0)
            st.divider()
            totals = st.columns(4)
            totals[0].metric("Responses", f"{len(results):,}")
            for panel, label in zip(totals[1:], ("positive", "neutral", "negative")):
                panel.metric(label.title(), f"{counts[label]:,}")
            skipped = len(results) - len(classified)
            if skipped:
                st.info(f"{skipped:,} entries could not be classified. Their status is included below and in the download.")
            if len(classified):
                chart, summary = st.columns([1.5, 1], gap="large")
                with chart:
                    st.markdown("**Sentiment distribution**")
                    distribution = pd.DataFrame({"Sentiment": ["Positive", "Neutral", "Negative"], "Responses": [counts["positive"], counts["neutral"], counts["negative"]]})
                    st.bar_chart(distribution, x="Sentiment", y="Responses", color="#17624B", horizontal=True, height=215, sort=False)
                with summary:
                    st.markdown("**A place to start**")
                    negative_share = counts["negative"] / len(classified)
                    st.write(f"{negative_share:.0%} of classified feedback is negative.")
                    st.caption("Review negative responses for recurring service issues. Sentiment alone does not identify a root cause.")
            filter_col, search_col, download_col = st.columns([1, 1.5, 1])
            selected = filter_col.selectbox("Show responses", ["All", "Negative", "Neutral", "Positive", "Unclassified"], key="sentiment_filter")
            search = search_col.text_input("Search feedback", placeholder="Search words or phrases…", key="feedback_search")
            visible = results if selected == "All" else results[results["sentiment"] == selected.lower()]
            if search.strip():
                visible = visible[visible["feedback"].str.contains(search.strip(), case=False, regex=False, na=False)]
            if visible.empty:
                st.info("No responses match these filters. Try another sentiment or search phrase.")
            st.dataframe(visible, hide_index=True, width="stretch", column_config={"feedback": st.column_config.TextColumn("Customer feedback", width="large"), "sentiment": "Sentiment", "status": "Status"})
            st.caption(f"Showing {len(visible):,} of {len(results):,} responses. The download includes all results and original CSV columns.")
            download_col.download_button("Download results", data=export_csv(results), file_name="feedback_sentiment.csv", mime="text/csv")
    st.download_button("Download sample CSV", data=(ROOT / "data" / "sample_feedback.csv").read_bytes(), file_name="sample_feedback.csv", mime="text/csv")


with model_tab:
    st.write("")
    st.subheader("A simple, measurable baseline")
    st.write("TF-IDF turns words and word pairs into numerical features. A linear Support Vector Machine uses those features to predict one of three sentiment classes.")
    a, b, c = st.columns(3)
    a.metric("Holdout accuracy", f"{metrics['accuracy']:.1%}")
    b.metric("Macro F1", f"{metrics['macro_f1']:.3f}")
    c.metric("Test responses", f"{metrics['test_rows']:,}")
    st.caption(f"80/20 stratified split · {metrics['train_rows']:,} training responses · Seed 42 · Majority baseline: {metrics['majority_baseline_accuracy']:.1%}")
    st.markdown("**Performance by sentiment**")
    report = pd.DataFrame({label.title(): metrics["classification_report"][label] for label in LABELS}).T
    report = report.rename(columns={"precision": "Precision", "recall": "Recall", "f1-score": "F1", "support": "Test rows"})
    st.dataframe(report, width="stretch", column_config={"Precision": st.column_config.NumberColumn(format="%.3f"), "Recall": st.column_config.NumberColumn(format="%.3f"), "F1": st.column_config.NumberColumn(format="%.3f"), "Test rows": st.column_config.NumberColumn(format="%d")})
    with st.expander("Confusion matrix"):
        matrix = pd.DataFrame(metrics["confusion_matrix"], index=[f"Actual {label}" for label in LABELS], columns=[f"Predicted {label}" for label in LABELS])
        st.dataframe(matrix, width="stretch")
    st.markdown("**Data & evaluation**")
    st.write(f"{metrics['dataset']}. After removing repeated or contradictory text, {metrics['usable_rows']:,} responses remain. The vectorizer learns only from the training split; the bundled model is the same model measured on the holdout.")
    st.caption("Dataset: [CrowdFlower on Kaggle](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment) · CC BY-NC-SA 4.0 · Educational, noncommercial baseline")
    st.markdown("**Where to be careful**")
    st.write("English airline feedback is the training domain. Sarcasm, mixed opinions, short messages, and unfamiliar wording can be misclassified. Neutral is a learned class, not a low-confidence fallback. The SVM does not provide a confidence percentage.")
    st.caption("Built with Python, Pandas, scikit-learn, TF-IDF, SVM, and Streamlit. Uploaded feedback is held in the current session and is not written to disk or sent to an external model API.")

st.markdown('<div class="footer">CUSTOMER FEEDBACK SENTIMENT ANALYSIS &nbsp; / &nbsp; Python · scikit-learn · Pandas · TF-IDF · SVM · Streamlit</div>', unsafe_allow_html=True)
