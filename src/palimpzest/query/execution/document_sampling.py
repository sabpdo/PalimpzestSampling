"""
Ordering of root-dataset records for sentinel (sample-based) operator evaluation.

Sentinel execution uses stable string keys ``{dataset_id}---{row_index}``. This module
centralizes how those keys are ordered before ``OpFrontier`` consumes them.
"""

from __future__ import annotations

from collections.abc import Callable
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
) -> list[str]:
    """
    Stratified sentinel document order — **implement here** (or use ``sample_documents(..., sampler=…)``).

    Called when ``sample_documents(..., method="stratified")``. Must return a permutation of
    ``f"{dataset_id}---{i}"`` for ``i`` in ``0 .. num_records - 1``.

    Parameters
    ----------
    dataset_id
        Root dataset id (same as for baseline shuffle).
    num_records
        Number of rows in that dataset.
    rng
        Seeded generator (use for any within-stratum randomness).
    num_strata
        Value from ``QueryProcessorConfig.stratified_num_strata`` / ``sample_documents(..., stratified_num_strata=…)``.
    """
    raise NotImplementedError(
        "Implement stratified_source_keys() in palimpzest/query/execution/document_sampling.py, "
        "or pass sample_documents(..., sampler=your_callable)."
    )


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
