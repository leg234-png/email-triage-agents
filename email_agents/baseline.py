"""Baseline supervisée : TF-IDF (mots + caractères) + régression logistique, une par étiquette."""
from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

TARGETS = ["intent", "priority", "sender"]


def text_of(df: pd.DataFrame) -> pd.Series:
    return df["sender_email"] + " " + df["subject"] + " " + df["body"]


def make_model() -> Pipeline:
    features = FeatureUnion([
        ("words", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
    ])
    return Pipeline([("tfidf", features), ("clf", LogisticRegression(max_iter=2000, C=4.0))])


class TfidfBaseline:
    def __init__(self):
        self.models = {t: make_model() for t in TARGETS}

    def fit(self, df: pd.DataFrame) -> "TfidfBaseline":
        X = text_of(df)
        for t, m in self.models.items():
            m.fit(X, df[t])
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = text_of(df)
        return pd.DataFrame({t: m.predict(X) for t, m in self.models.items()}, index=df.index)
