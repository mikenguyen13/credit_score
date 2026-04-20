"""
Shared helpers used throughout the book.

Kept small and dependency-light so every chapter can import it without
dragging optional packages.
"""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import requests

BOOK_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = BOOK_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


def _cache_get(url: str, filename: str, timeout: int = 60) -> Path:
    dst = DATA_DIR / filename
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    dst.write_bytes(resp.content)
    return dst


def load_german_credit() -> pd.DataFrame:
    """UCI Statlog German Credit Data (1994). Public domain.

    Returns a dataframe with 20 features plus target `default` (1=bad, 0=good).
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    path = _cache_get(url, "german.data")
    columns = [
        "status", "duration", "credit_history", "purpose", "amount",
        "savings", "employment", "installment_rate", "personal_status",
        "other_debtors", "residence_since", "property", "age",
        "other_installment", "housing", "existing_credits", "job",
        "people_liable", "telephone", "foreign_worker", "target",
    ]
    df = pd.read_csv(path, sep=" ", header=None, names=columns)
    df["default"] = (df["target"] == 2).astype(int)
    df = df.drop(columns=["target"])
    return df


def load_taiwan_default() -> pd.DataFrame:
    """UCI Default of Credit Card Clients (Yeh & Lien 2009). Public.

    30,000 Taiwanese credit-card customers with default payment next month.
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
    path = _cache_get(url, "taiwan_default.xls")
    df = pd.read_excel(path, header=1)
    df = df.rename(columns={"default payment next month": "default", "ID": "id"})
    return df


def load_home_credit_sample(n_rows: int | None = 50_000, seed: int = 0) -> pd.DataFrame:
    """Small application_train sample mirrored on GitHub (CC0 via Kaggle).

    For the full ~300k-row dataset, download from Kaggle and place into
    book/data/application_train.csv.
    """
    cached = DATA_DIR / "application_train.csv"
    if not cached.exists():
        url = (
            "https://raw.githubusercontent.com/sibyjackgrove/"
            "home-credit-default-risk/master/data/application_train.csv"
        )
        try:
            _cache_get(url, "application_train.csv")
        except Exception:
            raise FileNotFoundError(
                "Put application_train.csv (Kaggle Home Credit) into book/data/."
            )
    df = pd.read_csv(cached)
    if n_rows is not None and len(df) > n_rows:
        df = df.sample(n=n_rows, random_state=seed).reset_index(drop=True)
    df = df.rename(columns={"TARGET": "default"})
    return df


def train_valid_test_split(
    df: pd.DataFrame, y_col: str = "default",
    valid_size: float = 0.2, test_size: float = 0.2, seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_test = int(len(df) * test_size)
    n_valid = int(len(df) * valid_size)
    test = df.iloc[idx[:n_test]].reset_index(drop=True)
    valid = df.iloc[idx[n_test:n_test + n_valid]].reset_index(drop=True)
    train = df.iloc[idx[n_test + n_valid:]].reset_index(drop=True)
    return train, valid, test


def ks_statistic(y_true, y_score) -> float:
    """Kolmogorov-Smirnov statistic for a binary classifier."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    cum_pos = np.cumsum(y_sorted) / max(y_sorted.sum(), 1)
    cum_neg = np.cumsum(1 - y_sorted) / max((1 - y_sorted).sum(), 1)
    return float(np.max(np.abs(cum_pos - cum_neg)))


def gini(y_true, y_score) -> float:
    from sklearn.metrics import roc_auc_score
    return float(2 * roc_auc_score(y_true, y_score) - 1)


def psi(expected, actual, buckets: int = 10) -> float:
    """Population Stability Index."""
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    quantiles = np.quantile(expected, np.linspace(0, 1, buckets + 1))
    quantiles[0], quantiles[-1] = -np.inf, np.inf
    e, _ = np.histogram(expected, bins=quantiles)
    a, _ = np.histogram(actual, bins=quantiles)
    e = e / max(e.sum(), 1); a = a / max(a.sum(), 1)
    eps = 1e-6
    return float(np.sum((e - a) * np.log((e + eps) / (a + eps))))


def scorecard_points(prob, base_score: int = 600, base_odds: float = 50.0,
                     pdo: int = 20) -> np.ndarray:
    """Convert probability-of-default to points (higher = safer)."""
    prob = np.clip(np.asarray(prob), 1e-9, 1 - 1e-9)
    odds_good = (1 - prob) / prob
    factor = pdo / np.log(2.0)
    offset = base_score - factor * np.log(base_odds)
    return offset + factor * np.log(odds_good)


__all__ = [
    "BOOK_ROOT", "DATA_DIR",
    "load_german_credit", "load_taiwan_default", "load_home_credit_sample",
    "train_valid_test_split",
    "ks_statistic", "gini", "psi", "scorecard_points",
]
