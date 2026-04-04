import numpy as np
import pytest

from palimpzest.query.execution.document_sampling import (
    StratifiedDocumentSampler,
    sample_documents,
    stratified_source_keys,
)


def test_sample_documents_random_is_permutation():
    rng = np.random.default_rng(0)
    keys = sample_documents("ds", 100, rng, method="random")
    assert len(keys) == 100
    parsed = sorted(int(k.split("---")[1]) for k in keys)
    assert parsed == list(range(100))


def test_stratified_source_keys_is_unimplemented_stub():
    rng = np.random.default_rng(1)
    with pytest.raises(NotImplementedError, match="Implement stratified_source_keys"):
        stratified_source_keys("ds", 17, rng, num_strata=5)


def test_sample_documents_stratified_delegates_to_stub():
    rng = np.random.default_rng(1)
    with pytest.raises(NotImplementedError):
        sample_documents("ds", 17, rng, method="stratified", stratified_num_strata=5)


def test_stratified_document_sampler_class_stub():
    with pytest.raises(NotImplementedError, match="Subclass StratifiedDocumentSampler"):
        StratifiedDocumentSampler().order("ds", 5, np.random.default_rng(0), num_strata=2)


def test_custom_sampler_override():
    rng = np.random.default_rng(2)

    def rev(_ds: str, n: int, _rng: np.random.Generator) -> list[str]:
        return [f"x---{i}" for i in range(n - 1, -1, -1)]

    keys = sample_documents("ignored", 4, rng, method="random", sampler=rev)
    assert keys == ["x---3", "x---2", "x---1", "x---0"]
