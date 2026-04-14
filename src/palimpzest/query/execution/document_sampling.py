"""
Ordering of root-dataset records for sentinel (sample-based) operator evaluation.

Sentinel execution uses stable string keys ``{dataset_id}---{row_index}``. This module
centralizes how those keys are ordered before ``OpFrontier`` consumes them.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np

DocumentSamplingMethod = Literal["random", "stratified"]

# Optional override: return full sentinel source keys in the desired evaluation order.
DocumentSourceSampler = Callable[[str, int, np.random.Generator], list[str]]


def stratified_source_keys(
    dataset_id: str,
    num_records: int,
    rng: np.random.Generator,
    *,
    num_strata: int = 8,
    features_csv: Path | None = None,
) -> list[str]:
    """
    Feature-stratified sentinel order: shuffle within strata, then round-robin across strata.

    Strata come from ``scripts/extract_features.py`` feature columns (see
    :mod:`palimpzest.query.execution.feature_stratified_sampling`). Default CSV is
    ``<cwd>/papers/paper_features.csv`` or ``PALIMPZEST_STRATIFIED_FEATURES_PATH``.

    Called when ``sample_documents(..., method="stratified")``. Returns a permutation of
    ``f"{dataset_id}---{i}"`` for ``i`` in ``0 .. num_records - 1``.
    """
    from . import feature_stratified_sampling as fss

    row_order = fss.feature_stratified_row_order(
        num_records, rng, num_strata, features_csv=features_csv
    )
    return [f"{dataset_id}---{int(i)}" for i in row_order]


class StratifiedDocumentSampler:
    """
    Optional OOP hook: subclass and implement ``order``, then wire in via ``sampler``::

        class MySampler(StratifiedDocumentSampler):
            def order(self, dataset_id, num_records, rng, *, num_strata=8):
                ...

        sample_documents(..., sampler=lambda ds, n, r: MySampler().order(ds, n, r, num_strata=8))
    """

    def order(
        self,
        dataset_id: str,
        num_records: int,
        rng: np.random.Generator,
        *,
        num_strata: int = 8,
    ) -> list[str]:
        raise NotImplementedError("Subclass StratifiedDocumentSampler and implement order().")


class FeatureStratifiedDocumentSampler(StratifiedDocumentSampler):
    """Stratified ordering from a feature table (same policy as :func:`stratified_source_keys`)."""

    def __init__(self, features_csv: Path | None = None) -> None:
        self._features_csv = features_csv

    def order(
        self,
        dataset_id: str,
        num_records: int,
        rng: np.random.Generator,
        *,
        num_strata: int = 8,
    ) -> list[str]:
        return stratified_source_keys(
            dataset_id,
            num_records,
            rng,
            num_strata=num_strata,
            features_csv=self._features_csv,
        )


def sample_documents(
    dataset_id: str,
    num_records: int,
    rng: np.random.Generator,
    *,
    method: DocumentSamplingMethod = "random",
    stratified_num_strata: int = 8,
    sampler: DocumentSourceSampler | None = None,
) -> list[str]:
    """
    Build the ordered list of sentinel source-index keys for one root dataset.

    Parameters
    ----------
    dataset_id
        Root dataset identifier (embedded in each key).
    num_records
        Number of rows in the dataset (indices ``0 .. num_records - 1``).
    rng
        NumPy random generator (used by ``random`` / ``stratified`` and passed to ``sampler``).
    method
        ``random``: shuffle row indices (current Palimpzest baseline).
        ``stratified``: delegates to :func:`stratified_source_keys` (your implementation).
    stratified_num_strata
        Passed through to :func:`stratified_source_keys` when ``method="stratified"``.
    sampler
        If set, called as ``sampler(dataset_id, num_records, rng)`` and its return value
        is used instead of ``method`` (plug-in point for custom policies).
    """
    if sampler is not None:
        return sampler(dataset_id, num_records, rng)

    keys = [f"{dataset_id}---{int(idx)}" for idx in range(num_records)]

    if num_records == 0:
        return []

    if method == "random":
        rng.shuffle(keys)
        return keys

    if method == "stratified":
        return stratified_source_keys(
            dataset_id,
            num_records,
            rng,
            num_strata=stratified_num_strata,
        )

    raise ValueError(f"Unknown document sampling method: {method!r}")
