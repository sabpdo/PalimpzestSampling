#!/usr/bin/env python3
"""
Build seeded synthetic feature tables over real PDFs to stress-test stratified sampling.

The stratifier reads only ``paper_features.csv`` for ordering; PDF contents need not match
these rows. Copy (or symlink) PDFs from an existing corpus, then assign skewed features so
small budgets make uniform random ordering likely to under-cover rare strata.

Example::

    python scripts/generate_stratifier_stress_dataset.py \\
      --output-dir papers/stratifier_stress_s42 \\
      --seed 42 \\
      --n-docs 60 \\
      --preset rare_numeric_tail \\
      --source-papers papers

Advanced overrides (merge into defaults)::

    python scripts/generate_stratifier_stress_dataset.py ... \\
      --stress-params-json '{"tail_fraction": 0.2, "tail_domain": "cs"}'
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Must match feature_stratified_sampling.FEATURE_COLUMNS / extract_features output.
_FEATURE_COLUMNS = (
    "word_count",
    "section_count",
    "avg_sentence_length",
    "figure_count",
    "table_count",
    "complexity_score",
    "domain",
)


@dataclass
class StressDatasetParams:
    """Tunable skew knobs (defaults match original presets). Values are clamped in builders."""

    # rare_numeric_tail
    tail_fraction: float = 0.12
    bulk_word_low: int = 3500
    bulk_word_high: int = 9000
    tail_word_low: int = 82000
    tail_word_high: int = 118000
    tail_domain: str = "physics"
    bulk_domains: tuple[str, ...] = ("bio_medical", "cs", "math")

    # domain_heavy_skew
    majority_fraction: float = 0.92
    majority_domain: str = "bio_medical"
    minority_domain: str = "cs"

    # bimodal_sections
    short_section_fraction: float = 0.85
    short_section_low: int = 12
    short_section_high: int = 48
    long_section_low: int = 160
    long_section_high: int = 280
    bimodal_domains: tuple[str, ...] = ("bio_medical", "cs", "math", "physics")

    # ultra_rare_domain_tail
    ultra_rare_fraction: float = 0.03
    ultra_bulk_domains: tuple[str, ...] = ("bio_medical", "cs", "math")
    ultra_rare_domain: str = "physics"
    ultra_bulk_word_low: int = 2800
    ultra_bulk_word_high: int = 9000
    ultra_rare_word_low: int = 120000
    ultra_rare_word_high: int = 220000
    ultra_rare_section_low: int = 240
    ultra_rare_section_high: int = 520

    # adversarial_feature_conflict
    conflict_fraction: float = 0.45
    conflict_domain_a: str = "bio_medical"
    conflict_domain_b: str = "physics"
    conflict_a_word_low: int = 6000
    conflict_a_word_high: int = 22000
    conflict_b_word_low: int = 45000
    conflict_b_word_high: int = 115000

    @classmethod
    def defaults(cls) -> StressDatasetParams:
        return cls()

    @classmethod
    def from_dict(cls, d: dict) -> StressDatasetParams:
        base = asdict(cls.defaults())
        for k, v in d.items():
            if k not in base:
                continue
            if k in ("bulk_domains", "bimodal_domains", "ultra_bulk_domains"):
                if isinstance(v, str):
                    base[k] = (v,)
                elif isinstance(v, (list, tuple)):
                    base[k] = tuple(v)
                else:
                    base[k] = base[k]
            else:
                base[k] = v
        return cls(**base)

    def to_json_serializable(self) -> dict:
        d = asdict(self)
        d["bulk_domains"] = list(self.bulk_domains)
        d["bimodal_domains"] = list(self.bimodal_domains)
        d["ultra_bulk_domains"] = list(self.ultra_bulk_domains)
        return d


def merge_stress_params(current: StressDatasetParams, overrides: dict) -> StressDatasetParams:
    """Shallow-merge ``overrides`` into ``current`` (e.g. partial JSON)."""
    base = current.to_json_serializable()
    base.update(overrides)
    return StressDatasetParams.from_dict(base)


PRESET_DESCRIPTIONS = {
    "rare_numeric_tail": (
        "Majority in a tight numeric ``word_count`` band; minority in a distant high-count tail. "
        "Random ordering often delays the tail at low budgets; stratified spreads mass across strata."
    ),
    "domain_heavy_skew": (
        "One dominant ``domain`` with homogeneous numerics; a rare second domain with shifted features."
    ),
    "bimodal_sections": (
        "Two clusters on ``section_count`` (short vs long papers) with configurable imbalance."
    ),
    "ultra_rare_domain_tail": (
        "Needle-in-haystack setup: a tiny minority (<5% default) with extreme length/structure stats "
        "and rare domain labels."
    ),
    "adversarial_feature_conflict": (
        "Two conflicting regimes where feature signals disagree (e.g., high word_count with high complexity "
        "vs lower counts with moderate complexity), creating harder random coverage at low budgets."
    ),
}


def _iter_pdf_paths(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(root.rglob("*.pdf"))


def _pick_sources(source_root: Path, n: int, rng: random.Random) -> list[Path]:
    pool = _iter_pdf_paths(source_root)
    if not pool:
        raise FileNotFoundError(f"No PDFs under {source_root}")
    return [pool[rng.randrange(len(pool))] for _ in range(n)]


def _clamp01(x: float) -> float:
    return max(0.01, min(0.99, float(x)))


def _make_rows_rare_numeric_tail(rng: random.Random, n: int, p: StressDatasetParams) -> list[dict]:
    tf = _clamp01(p.tail_fraction)
    bulk_domains = tuple(p.bulk_domains) if p.bulk_domains else ("bio_medical", "cs", "math")
    rows: list[dict] = []
    bw_lo, bw_hi = sorted((int(p.bulk_word_low), int(p.bulk_word_high)))
    tw_lo, tw_hi = sorted((int(p.tail_word_low), int(p.tail_word_high)))
    if bw_lo == bw_hi:
        bw_hi = bw_lo + 1
    if tw_lo == tw_hi:
        tw_hi = tw_lo + 1
    for _ in range(n):
        if rng.random() < tf:
            word_count = rng.randint(tw_lo, tw_hi)
            section_count = rng.randint(140, 260)
            avg_sentence_length = round(rng.uniform(16.0, 28.0), 2)
            figure_count = rng.randint(15, 45)
            table_count = rng.randint(2, 14)
            complexity_score = round(rng.uniform(0.52, 0.66), 4)
            domain = str(p.tail_domain) if p.tail_domain else "physics"
        else:
            word_count = rng.randint(bw_lo, bw_hi)
            section_count = rng.randint(18, 95)
            avg_sentence_length = round(rng.uniform(13.0, 22.0), 2)
            figure_count = rng.randint(0, 12)
            table_count = rng.randint(0, 6)
            complexity_score = round(rng.uniform(0.34, 0.46), 4)
            domain = rng.choice(bulk_domains)
        rows.append(
            {
                "word_count": word_count,
                "section_count": section_count,
                "avg_sentence_length": avg_sentence_length,
                "figure_count": figure_count,
                "table_count": table_count,
                "complexity_score": complexity_score,
                "domain": domain,
            }
        )
    return rows


def _make_rows_domain_heavy_skew(rng: random.Random, n: int, p: StressDatasetParams) -> list[dict]:
    mf = _clamp01(p.majority_fraction)
    maj_dom = str(p.majority_domain) if p.majority_domain else "bio_medical"
    min_dom = str(p.minority_domain) if p.minority_domain else "cs"
    rows: list[dict] = []
    for _ in range(n):
        if rng.random() < mf:
            domain = maj_dom
            word_count = rng.randint(6_000, 14_000)
            section_count = rng.randint(40, 110)
            avg_sentence_length = round(rng.uniform(15.0, 21.0), 2)
            figure_count = rng.randint(2, 18)
            table_count = rng.randint(0, 8)
            complexity_score = round(rng.uniform(0.37, 0.47), 4)
        else:
            domain = min_dom
            word_count = rng.randint(28_000, 48_000)
            section_count = rng.randint(10, 55)
            avg_sentence_length = round(rng.uniform(11.0, 17.5), 2)
            figure_count = rng.randint(0, 8)
            table_count = rng.randint(0, 4)
            complexity_score = round(rng.uniform(0.42, 0.58), 4)
        rows.append(
            {
                "word_count": word_count,
                "section_count": section_count,
                "avg_sentence_length": avg_sentence_length,
                "figure_count": figure_count,
                "table_count": table_count,
                "complexity_score": complexity_score,
                "domain": domain,
            }
        )
    return rows


def _make_rows_bimodal_sections(rng: random.Random, n: int, p: StressDatasetParams) -> list[dict]:
    sf = _clamp01(p.short_section_fraction)
    ss_lo, ss_hi = sorted((int(p.short_section_low), int(p.short_section_high)))
    ls_lo, ls_hi = sorted((int(p.long_section_low), int(p.long_section_high)))
    if ss_lo == ss_hi:
        ss_hi = ss_lo + 1
    if ls_lo == ls_hi:
        ls_hi = ls_lo + 1
    doms = tuple(p.bimodal_domains) if p.bimodal_domains else ("bio_medical", "cs", "math", "physics")
    rows: list[dict] = []
    for _ in range(n):
        if rng.random() < sf:
            section_count = rng.randint(ss_lo, ss_hi)
            word_count = rng.randint(5_000, 18_000)
        else:
            section_count = rng.randint(ls_lo, ls_hi)
            word_count = rng.randint(35_000, 95_000)
        avg_sentence_length = round(rng.uniform(13.0, 24.0), 2)
        figure_count = rng.randint(0, 22)
        table_count = rng.randint(0, 10)
        complexity_score = round(rng.uniform(0.35, 0.52), 4)
        domain = rng.choice(doms)
        rows.append(
            {
                "word_count": word_count,
                "section_count": section_count,
                "avg_sentence_length": avg_sentence_length,
                "figure_count": figure_count,
                "table_count": table_count,
                "complexity_score": complexity_score,
                "domain": domain,
            }
        )
    return rows


def _make_rows_ultra_rare_domain_tail(rng: random.Random, n: int, p: StressDatasetParams) -> list[dict]:
    rf = _clamp01(p.ultra_rare_fraction)
    bulk_domains = tuple(p.ultra_bulk_domains) if p.ultra_bulk_domains else ("bio_medical", "cs", "math")
    bw_lo, bw_hi = sorted((int(p.ultra_bulk_word_low), int(p.ultra_bulk_word_high)))
    rw_lo, rw_hi = sorted((int(p.ultra_rare_word_low), int(p.ultra_rare_word_high)))
    rs_lo, rs_hi = sorted((int(p.ultra_rare_section_low), int(p.ultra_rare_section_high)))
    if bw_lo == bw_hi:
        bw_hi = bw_lo + 1
    if rw_lo == rw_hi:
        rw_hi = rw_lo + 1
    if rs_lo == rs_hi:
        rs_hi = rs_lo + 1

    rows: list[dict] = []
    for _ in range(n):
        if rng.random() < rf:
            rows.append(
                {
                    "word_count": rng.randint(rw_lo, rw_hi),
                    "section_count": rng.randint(rs_lo, rs_hi),
                    "avg_sentence_length": round(rng.uniform(17.5, 31.0), 2),
                    "figure_count": rng.randint(16, 52),
                    "table_count": rng.randint(3, 18),
                    "complexity_score": round(rng.uniform(0.63, 0.84), 4),
                    "domain": str(p.ultra_rare_domain) if p.ultra_rare_domain else "physics",
                }
            )
        else:
            rows.append(
                {
                    "word_count": rng.randint(bw_lo, bw_hi),
                    "section_count": rng.randint(10, 80),
                    "avg_sentence_length": round(rng.uniform(12.0, 20.0), 2),
                    "figure_count": rng.randint(0, 10),
                    "table_count": rng.randint(0, 6),
                    "complexity_score": round(rng.uniform(0.30, 0.45), 4),
                    "domain": rng.choice(bulk_domains),
                }
            )
    return rows


def _make_rows_adversarial_feature_conflict(rng: random.Random, n: int, p: StressDatasetParams) -> list[dict]:
    cf = _clamp01(p.conflict_fraction)
    a_w_lo, a_w_hi = sorted((int(p.conflict_a_word_low), int(p.conflict_a_word_high)))
    b_w_lo, b_w_hi = sorted((int(p.conflict_b_word_low), int(p.conflict_b_word_high)))
    if a_w_lo == a_w_hi:
        a_w_hi = a_w_lo + 1
    if b_w_lo == b_w_hi:
        b_w_hi = b_w_lo + 1

    rows: list[dict] = []
    for _ in range(n):
        if rng.random() < cf:
            # Cohort A: shorter-ish documents but unexpectedly high complexity.
            rows.append(
                {
                    "word_count": rng.randint(a_w_lo, a_w_hi),
                    "section_count": rng.randint(14, 66),
                    "avg_sentence_length": round(rng.uniform(10.0, 16.8), 2),
                    "figure_count": rng.randint(0, 8),
                    "table_count": rng.randint(0, 4),
                    "complexity_score": round(rng.uniform(0.64, 0.86), 4),
                    "domain": str(p.conflict_domain_a) if p.conflict_domain_a else "bio_medical",
                }
            )
        else:
            # Cohort B: long documents but moderate complexity; opposite signal profile.
            rows.append(
                {
                    "word_count": rng.randint(b_w_lo, b_w_hi),
                    "section_count": rng.randint(96, 260),
                    "avg_sentence_length": round(rng.uniform(18.0, 30.0), 2),
                    "figure_count": rng.randint(10, 38),
                    "table_count": rng.randint(2, 14),
                    "complexity_score": round(rng.uniform(0.32, 0.52), 4),
                    "domain": str(p.conflict_domain_b) if p.conflict_domain_b else "physics",
                }
            )
    return rows


def _dispatch_rows(preset: str, rng: random.Random, n: int, params: StressDatasetParams) -> list[dict]:
    if preset == "rare_numeric_tail":
        return _make_rows_rare_numeric_tail(rng, n, params)
    if preset == "domain_heavy_skew":
        return _make_rows_domain_heavy_skew(rng, n, params)
    if preset == "bimodal_sections":
        return _make_rows_bimodal_sections(rng, n, params)
    if preset == "ultra_rare_domain_tail":
        return _make_rows_ultra_rare_domain_tail(rng, n, params)
    if preset == "adversarial_feature_conflict":
        return _make_rows_adversarial_feature_conflict(rng, n, params)
    raise ValueError(f"Unknown preset: {preset}")


AVAILABLE_PRESETS: tuple[str, ...] = (
    "adversarial_feature_conflict",
    "bimodal_sections",
    "domain_heavy_skew",
    "rare_numeric_tail",
    "ultra_rare_domain_tail",
)


def generate_stratifier_stress_dataset(
    repo_root: Path,
    *,
    output_dir: str,
    seed: int,
    n_docs: int,
    preset: str,
    source_papers: str,
    link_mode: str = "copy",
    stress_params: StressDatasetParams | None = None,
) -> dict:
    """
    Populate ``output_dir`` under repo_root with ``doc_XXXX.pdf`` and ``paper_features.csv``.

    Returns metadata dict including absolute paths and skew summary statistics.
    """
    if preset not in AVAILABLE_PRESETS:
        raise ValueError(f"Unknown preset {preset!r}; choose from {sorted(AVAILABLE_PRESETS)}")
    if link_mode not in {"copy", "symlink"}:
        raise ValueError("link_mode must be 'copy' or 'symlink'")

    params = stress_params if stress_params is not None else StressDatasetParams.defaults()

    out_rel = output_dir.strip().strip("/")
    if not out_rel or ".." in Path(out_rel).parts:
        raise ValueError("output_dir must be a relative path inside the repo without '..'")

    repo_root = repo_root.resolve()
    out_abs = (repo_root / out_rel).resolve()
    if not str(out_abs).startswith(str(repo_root)):
        raise ValueError("output_dir must resolve inside repo_root")

    source_root = (repo_root / source_papers).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source papers directory missing: {source_root}")

    rng = random.Random(seed)
    sources = _pick_sources(source_root, n_docs, rng)
    feature_dicts = _dispatch_rows(preset, rng, n_docs, params)

    out_abs.mkdir(parents=True, exist_ok=True)
    for p in out_abs.glob("doc_*.pdf"):
        p.unlink()

    rel_paths: list[str] = []
    for i, src in enumerate(sources):
        dest = out_abs / f"doc_{i:04d}.pdf"
        if link_mode == "copy":
            shutil.copy2(src, dest)
        else:
            if dest.exists() or dest.is_symlink():
                dest.unlink()
            dest.symlink_to(src.resolve())
        rel_paths.append(str(Path(out_rel) / dest.name))

    records = []
    for row_idx, (path_str, feats) in enumerate(zip(rel_paths, feature_dicts)):
        row = {"row_index": row_idx, "path": path_str.replace("\\", "/")}
        for col in _FEATURE_COLUMNS:
            row[col] = feats[col]
        records.append(row)

    csv_path = out_abs / "paper_features.csv"
    df = pd.DataFrame.from_records(records)
    df.to_csv(csv_path, index=False)

    summary = {
        "preset": preset,
        "preset_description": PRESET_DESCRIPTIONS[preset],
        "seed": seed,
        "n_docs": n_docs,
        "link_mode": link_mode,
        "papers_dir": str(out_abs),
        "papers_dir_rel": out_rel,
        "features_csv": str(csv_path),
        "features_csv_rel": str(Path(out_rel) / "paper_features.csv"),
        "domain_counts": df["domain"].value_counts().to_dict(),
        "word_count_quantiles": df["word_count"].quantile([0.05, 0.5, 0.95]).to_dict(),
        "stress_params": params.to_json_serializable(),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=_REPO_ROOT,
        help="Repository root (default: parent of scripts/).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory under repo root, e.g. papers/stratifier_stress_s42",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-docs", type=int, default=48)
    parser.add_argument(
        "--preset",
        choices=list(AVAILABLE_PRESETS),
        default="rare_numeric_tail",
    )
    parser.add_argument(
        "--source-papers",
        default="papers",
        help="Existing PDF tree to sample from (default: papers).",
    )
    parser.add_argument(
        "--link-mode",
        choices=["copy", "symlink"],
        default="copy",
        help="copy: duplicate PDF bytes; symlink: faster, keeps corpus path stable.",
    )
    parser.add_argument(
        "--stress-params-json",
        type=Path,
        default=None,
        help="Optional JSON file with StressDatasetParams fields to merge over defaults.",
    )
    parser.add_argument(
        "--json-summary",
        action="store_true",
        help="Print machine-readable summary JSON on stdout.",
    )
    args = parser.parse_args()

    stress_params: StressDatasetParams | None = None
    if args.stress_params_json is not None:
        raw = json.loads(args.stress_params_json.read_text(encoding="utf-8"))
        stress_params = StressDatasetParams.from_dict(raw)

    try:
        summary = generate_stratifier_stress_dataset(
            args.repo_root.resolve(),
            output_dir=args.output_dir,
            seed=args.seed,
            n_docs=args.n_docs,
            preset=args.preset,
            source_papers=args.source_papers,
            link_mode=args.link_mode,
            stress_params=stress_params,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    if args.json_summary:
        wcq = summary["word_count_quantiles"]
        summary_out = {
            **summary,
            "word_count_quantiles": {str(k): float(v) for k, v in wcq.items()},
            "domain_counts": {str(k): int(v) for k, v in summary["domain_counts"].items()},
        }
        print(json.dumps(summary_out, indent=2))
    else:
        print(f"Wrote {summary['n_docs']} PDFs under {summary['papers_dir']}")
        print(f"Features CSV: {summary['features_csv']}")
        print(f"Preset {summary['preset']}: {summary['preset_description']}")
        print(f"Domain counts: {summary['domain_counts']}")

    print(f"STRAT_BENCH_FEATURES_CSV:{summary['features_csv_rel']}")
    print(f"STRAT_BENCH_PAPERS_DIR:{summary['papers_dir_rel']}")


if __name__ == "__main__":
    main()
