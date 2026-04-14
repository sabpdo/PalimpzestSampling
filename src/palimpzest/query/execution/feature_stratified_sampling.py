"""
Feature-table stratification for sentinel document order.

Expects a CSV produced by ``scripts/extract_features.py --scan`` with columns
``row_index``, ``path``, and the numeric features listed in ``FEATURE_COLUMNS``.
Set ``PALIMPZEST_STRATIFIED_FEATURES_PATH`` to override the default ``<cwd>/papers/paper_features.csv``.
"""

from __future__ import annotations

import os
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

# Keep in sync with ``STRATIFICATION_FEATURE_FIELDS`` in scripts/extract_features.py.
FEATURE_COLUMNS = (
    "word_count",
    "section_count",
    "avg_sentence_length",
    "figure_count",
    "table_count",
    "complexity_score",
)


def default_features_csv_path() -> Path:
    env = os.environ.get("PALIMPZEST_STRATIFIED_FEATURES_PATH")
    if env:
        return Path(env).expanduser()
    return Path.cwd() / "papers" / "paper_features.csv"


def _validate_feature_csv_columns(df: pd.DataFrame, csv_path: Path) -> None:
    for col in ("row_index", *FEATURE_COLUMNS):
        if col not in df.columns:
            raise ValueError(f"CSV {csv_path} missing required column {col!r}")


def read_feature_table(csv_path: Path, *, max_rows: int | None = None) -> pd.DataFrame:
    """Load and validate a feature CSV; optional cap on the number of rows (after sorting by ``row_index``)."""
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Stratified sampling CSV not found: {csv_path}. "
            "Run ``python scripts/extract_features.py --scan ./papers -o papers/paper_features.csv`` "
            "or set PALIMPZEST_STRATIFIED_FEATURES_PATH."
        )
    df = pd.read_csv(csv_path)
    _validate_feature_csv_columns(df, csv_path)
    df = df.sort_values("row_index").reset_index(drop=True)
    if max_rows is not None:
        df = df.iloc[:max_rows].copy()
    return df


def load_feature_frame(csv_path: Path, num_records: int) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Stratified sampling CSV not found: {csv_path}. "
            "Run ``python scripts/extract_features.py --scan ./papers -o papers/paper_features.csv`` "
            "or set PALIMPZEST_STRATIFIED_FEATURES_PATH."
        )
    df = pd.read_csv(csv_path)
    _validate_feature_csv_columns(df, csv_path)
    df = df.sort_values("row_index").reset_index(drop=True)
    if len(df) < num_records:
        raise ValueError(
            f"CSV {csv_path} has {len(df)} rows but num_records={num_records}; rebuild the feature table."
        )
    return df.iloc[:num_records].copy()


def composite_strata(features: np.ndarray, num_strata: int) -> np.ndarray:
    """
    Map each row to an integer stratum in ``0 .. num_strata - 1`` using mean
    marginal percentile ranks across ``FEATURE_COLUMNS`` (balanced buckets on
    the composite score when the empirical distribution is smooth).
    """
    n, d = features.shape
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    if d != len(FEATURE_COLUMNS):
        raise ValueError(f"Expected {len(FEATURE_COLUMNS)} feature columns, got {d}")
    k = max(1, min(int(num_strata), n))
    pct = np.empty((n, d), dtype=np.float64)
    for j in range(d):
        col = pd.Series(features[:, j])
        pct[:, j] = col.rank(pct=True, method="average").to_numpy(dtype=np.float64)
    comp = pct.mean(axis=1)
    raw = (comp * k).astype(np.int64)
    return np.clip(raw, 0, k - 1)


def round_robin_merge(strata: np.ndarray, rng: np.random.Generator) -> list[int]:
    """Shuffle within each stratum, then round-robin across strata for even interleaving."""
    n = len(strata)
    if n == 0:
        return []
    k = int(strata.max()) + 1
    buckets: list[list[int]] = [[] for _ in range(k)]
    for idx, s in enumerate(strata.tolist()):
        buckets[int(s)].append(idx)
    for b in buckets:
        rng.shuffle(b)
    queues = [deque(b) for b in buckets]
    order: list[int] = []
    while any(queues):
        for q in queues:
            if q:
                order.append(q.popleft())
    return order


def feature_stratified_row_order(
    num_records: int,
    rng: np.random.Generator,
    num_strata: int,
    *,
    features_csv: Path | None = None,
) -> list[int]:
    """Return a permutation of ``0 .. num_records - 1`` (dataset row indices)."""
    path = features_csv if features_csv is not None else default_features_csv_path()
    df = load_feature_frame(path, num_records)
    mat = df.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    strata = composite_strata(mat, num_strata)
    return round_robin_merge(strata, rng)


def feature_strata_per_index(
    num_records: int,
    num_strata: int,
    *,
    features_csv: Path | None = None,
) -> np.ndarray:
    """Stratum id for each row index (same assignment as :func:`feature_stratified_row_order`)."""
    path = features_csv if features_csv is not None else default_features_csv_path()
    df = load_feature_frame(path, num_records)
    mat = df.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    return composite_strata(mat, num_strata)
