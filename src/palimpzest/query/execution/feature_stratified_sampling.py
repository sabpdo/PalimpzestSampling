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
    "domain",
)
_MODE_COMPOSITE = "composite"
_MODE_SINGLE = "single"
_MODE_CARTESIAN = "cartesian"


def default_features_csv_path() -> Path:
    env = os.environ.get("PALIMPZEST_STRATIFIED_FEATURES_PATH")
    if env:
        return Path(env).expanduser()
    return Path.cwd() / "papers" / "paper_features.csv"


def _configured_feature_columns() -> tuple[str, ...]:
    env = os.environ.get("PALIMPZEST_STRATIFIED_FEATURE_COLUMNS", "").strip()
    if not env:
        return FEATURE_COLUMNS
    columns = tuple([c.strip() for c in env.split(",") if c.strip()])
    if not columns:
        raise ValueError("PALIMPZEST_STRATIFIED_FEATURE_COLUMNS is set but empty after parsing.")
    return columns


def _configured_mode() -> str:
    mode = os.environ.get("PALIMPZEST_STRATIFIED_MODE", _MODE_CARTESIAN).strip().lower()
    if mode not in {_MODE_COMPOSITE, _MODE_SINGLE, _MODE_CARTESIAN}:
        raise ValueError(
            f"Unsupported PALIMPZEST_STRATIFIED_MODE={mode!r}; "
            f"expected '{_MODE_COMPOSITE}', '{_MODE_SINGLE}', or '{_MODE_CARTESIAN}'."
        )
    return mode


def _configured_single_feature() -> str | None:
    value = os.environ.get("PALIMPZEST_STRATIFIED_SINGLE_FEATURE", "").strip()
    return value or None


def _validate_feature_csv_columns(df: pd.DataFrame, csv_path: Path, feature_columns: tuple[str, ...]) -> None:
    for col in ("row_index", *feature_columns):
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
    _validate_feature_csv_columns(df, csv_path, FEATURE_COLUMNS)
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
    feature_columns = _configured_feature_columns()
    _validate_feature_csv_columns(df, csv_path, feature_columns)
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
    if d < 1:
        raise ValueError("Expected at least one feature column for composite stratification.")
    k = max(1, min(int(num_strata), n))
    pct = np.empty((n, d), dtype=np.float64)
    for j in range(d):
        col = pd.Series(features[:, j])
        pct[:, j] = col.rank(pct=True, method="average").to_numpy(dtype=np.float64)
    comp = pct.mean(axis=1)
    raw = (comp * k).astype(np.int64)
    return np.clip(raw, 0, k - 1)


def single_feature_strata(feature_values: np.ndarray, num_strata: int) -> np.ndarray:
    """Map one feature vector to strata via percentile bins."""
    n = len(feature_values)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    k = max(1, min(int(num_strata), n))
    pct = pd.Series(feature_values).rank(pct=True, method="average").to_numpy(dtype=np.float64)
    raw = (pct * k).astype(np.int64)
    return np.clip(raw, 0, k - 1)


def cartesian_strata(df: pd.DataFrame, feature_columns: tuple[str, ...], num_strata: int) -> np.ndarray:
    """
    Assign each row to a bucket from the Cartesian product of per-feature bins.

    - Numeric features: binned into ``num_strata`` quantile buckets.
    - Categorical features (e.g. domain): each unique value is its own bin.

    With two features having 4 and 3 bins respectively, there are up to 12 buckets.
    Empty bucket combinations are remapped so IDs are contiguous.
    """
    n = len(df)
    if n == 0:
        return np.zeros(0, dtype=np.int64)

    per_feature_bins: list[np.ndarray] = []
    per_feature_sizes: list[int] = []

    for col in feature_columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            k = max(1, min(int(num_strata), n))
            try:
                bins = pd.qcut(series, q=k, labels=False, duplicates="drop")
            except ValueError:
                bins = pd.Series(np.zeros(n, dtype=int), index=series.index)
            bins = bins.fillna(0).astype(int)
            n_bins = int(bins.max()) + 1
        else:
            codes, uniques = pd.factorize(series.astype(str), sort=True)
            bins = pd.Series(codes, index=series.index)
            n_bins = len(uniques)
        per_feature_bins.append(bins.to_numpy(dtype=np.int64))
        per_feature_sizes.append(n_bins)

    # encode as a single integer using mixed-radix representation
    strata = np.zeros(n, dtype=np.int64)
    stride = 1
    for bins_arr, size in zip(reversed(per_feature_bins), reversed(per_feature_sizes)):
        strata += bins_arr * stride
        stride *= size

    # remap to contiguous 0..k-1 (empty combinations are dropped)
    _, strata = np.unique(strata, return_inverse=True)
    return strata.astype(np.int64)


def _column_to_numeric(series: pd.Series) -> np.ndarray:
    """Convert numeric/categorical feature columns to float arrays for ranking."""
    if pd.api.types.is_numeric_dtype(series):
        return series.to_numpy(dtype=np.float64)
    codes, _ = pd.factorize(series.astype(str), sort=True)
    return codes.astype(np.float64)


def _feature_matrix(df: pd.DataFrame, feature_columns: tuple[str, ...]) -> np.ndarray:
    cols = [_column_to_numeric(df[c]) for c in feature_columns]
    return np.column_stack(cols) if cols else np.empty((len(df), 0), dtype=np.float64)


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
    feature_columns = _configured_feature_columns()
    mode = _configured_mode()
    single_feature = _configured_single_feature()

    if mode == _MODE_SINGLE:
        chosen_feature = single_feature if single_feature is not None else feature_columns[0]
        if chosen_feature not in feature_columns:
            raise ValueError(
                f"PALIMPZEST_STRATIFIED_SINGLE_FEATURE={chosen_feature!r} must be in selected columns {feature_columns}."
            )
        vals = _column_to_numeric(df.loc[:, chosen_feature])
        strata = single_feature_strata(vals, num_strata)
    elif mode == _MODE_CARTESIAN:
        strata = cartesian_strata(df, feature_columns, num_strata)
    else:
        mat = _feature_matrix(df, feature_columns)
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
    feature_columns = _configured_feature_columns()
    mode = _configured_mode()
    single_feature = _configured_single_feature()
    if mode == _MODE_SINGLE:
        chosen_feature = single_feature if single_feature is not None else feature_columns[0]
        vals = _column_to_numeric(df.loc[:, chosen_feature])
        return single_feature_strata(vals, num_strata)
    mat = _feature_matrix(df, feature_columns)
    return composite_strata(mat, num_strata)
