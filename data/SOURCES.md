# Dataset provenance

The model uses **Twitter US Airline Sentiment**, released by CrowdFlower through its Data for Everyone collection and shared on Kaggle. The original file has 14,640 airline-related tweets from February 2015 with positive, neutral, and negative annotations.

- Original dataset: https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment
- Original conversion repository: https://github.com/benhamner/crowdflower-airline-twitter-sentiment
- Dataset card documenting its origin and license: https://huggingface.co/datasets/osanseviero/twitter-airline-sentiment
- Dataset license: **Creative Commons Attribution–NonCommercial–ShareAlike 4.0** — https://creativecommons.org/licenses/by-nc-sa/4.0/

`feedback.csv.gz` is an adapted copy retaining only `text` and `sentiment`. Text is lowercased, HTML entities are decoded, Unicode is normalized, and handles, URLs, retweet markers, and repeated whitespace are removed. User profile, location, tweet ID, and other metadata columns are not redistributed. This adaptation is provided under the same CC BY-NC-SA 4.0 terms; credit for the underlying data belongs to CrowdFlower and the original contributors. No endorsement is implied. The bundled trained artifact is supplied as part of this educational, noncommercial demonstration under the same dataset restrictions.

`train.py` additionally removes normalized-text duplicates and conflicting annotations before the stratified train/test split. Data preparation is deterministic. The retained training-data SHA-256 is recorded in `artifacts/metrics.json`.

`sample_feedback.csv` contains 12 newly written illustrative examples for the interface. It is **not** the training set or the holdout test set, and its results are not used to calculate the reported metrics.

This is an English airline-feedback baseline. Its measurements do not establish accuracy on retail reviews, other industries, languages, or live company data.
