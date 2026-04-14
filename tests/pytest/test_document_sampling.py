from pathlib import Path

import numpy as np
import pytest

from palimpzest.query.execution.document_sampling import (
    FeatureStratifiedDocumentSampler,
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


def _minimal_features_csv(path: Path, n: int) -> None:
    rows = ["row_index,path,word_count,section_count,avg_sentence_length,figure_count,table_count,complexity_score"]
    for i in range(n):
        rows.append(
            f"{i},p{i}.pdf,{10 + i},{2 + i % 3},{15.0 + 0.1 * i},{i % 4},{i % 2},{0.4 + 0.001 * i}"
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_stratified_source_keys_is_permutation(tmp_path: Path):
    csv_path = tmp_path / "feats.csv"
    _minimal_features_csv(csv_path, 17)
    rng = np.random.default_rng(1)
    keys = stratified_source_keys("ds", 17, rng, num_strata=5, features_csv=csv_path)
    assert len(keys) == 17
    parsed = sorted(int(k.split("---")[1]) for k in keys)
    assert parsed == list(range(17))
    assert all(k.startswith("ds---") for k in keys)


def test_sample_documents_stratified_matches_hook(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    csv_path = tmp_path / "feats.csv"
    _minimal_features_csv(csv_path, 12)
    monkeypatch.setenv("PALIMPZEST_STRATIFIED_FEATURES_PATH", str(csv_path))
    rng = np.random.default_rng(3)
    keys = sample_documents("x", 12, rng, method="stratified", stratified_num_strata=4)
    rng2 = np.random.default_rng(3)
    assert keys == stratified_source_keys("x", 12, rng2, num_strata=4, features_csv=csv_path)


def test_sample_documents_stratified_uses_cwd_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    csv_path = tmp_path / "papers" / "paper_features.csv"
    csv_path.parent.mkdir(parents=True)
    _minimal_features_csv(csv_path, 8)
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(0)
    keys = sample_documents("d", 8, rng, method="stratified", stratified_num_strata=3)
    assert len(keys) == 8
    assert sorted(int(k.split("---")[1]) for k in keys) == list(range(8))


def test_feature_stratified_document_sampler_matches_stratified_source_keys(tmp_path: Path):
    csv_path = tmp_path / "feats.csv"
    _minimal_features_csv(csv_path, 14)
    rng_a = np.random.default_rng(9)
    rng_b = np.random.default_rng(9)
    sampler = FeatureStratifiedDocumentSampler(features_csv=csv_path)
    assert sampler.order("q", 14, rng_a, num_strata=6) == stratified_source_keys(
        "q", 14, rng_b, num_strata=6, features_csv=csv_path
    )


def test_stratified_document_sampler_base_still_abstract():
    with pytest.raises(NotImplementedError, match="Subclass StratifiedDocumentSampler"):
        StratifiedDocumentSampler().order("ds", 5, np.random.default_rng(0), num_strata=2)


def test_custom_sampler_override():
    rng = np.random.default_rng(2)

    def rev(_ds: str, n: int, _rng: np.random.Generator) -> list[str]:
        return [f"x---{i}" for i in range(n - 1, -1, -1)]

    keys = sample_documents("ignored", 4, rng, method="random", sampler=rev)
    assert keys == ["x---3", "x---2", "x---1", "x---0"]
