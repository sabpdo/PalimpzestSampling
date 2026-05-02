#!/usr/bin/env python3
"""Concatenate two stress sweep output dirs (parallel shards) into one combined dir."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shard-a", type=Path, required=True)
    p.add_argument("--shard-b", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    a, b, out = args.shard_a.resolve(), args.shard_b.resolve(), args.out.resolve()
    for label, root in ("shard-a", a), ("shard-b", b):
        if not (root / "stress_dataset_manifest.csv").is_file():
            raise FileNotFoundError(f"{label}: missing stress_dataset_manifest.csv at {root}")

    out.mkdir(parents=True, exist_ok=True)
    (out / "pools").mkdir(exist_ok=True)
    (out / "runs").mkdir(exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)

    for shard in (a, b):
        for csv in (shard / "runs").glob("*_ab.csv"):
            shutil.copy2(csv, out / "runs" / csv.name)
        if (shard / "pools").is_dir():
            for sub in (shard / "pools").iterdir():
                if sub.is_dir():
                    dest = out / "pools" / sub.name
                    if dest.exists():
                        continue
                    shutil.copytree(sub, dest, symlinks=True)

    m_a = pd.read_csv(a / "stress_dataset_manifest.csv")
    m_b = pd.read_csv(b / "stress_dataset_manifest.csv")
    pd.concat([m_a, m_b], ignore_index=True).drop_duplicates(subset=["dataset_id"]).sort_values(
        "dataset_id"
    ).to_csv(out / "stress_dataset_manifest.csv", index=False)

    runs_a = list((a / "runs").glob("*_ab.csv"))
    runs_b = list((b / "runs").glob("*_ab.csv"))
    pieces = []
    for path in runs_a + runs_b:
        pieces.append(pd.read_csv(path))
    big = pd.concat(pieces, ignore_index=True)
    big.to_csv(out / "stress_ab_all_runs.csv", index=False)

    meta = {
        "merged_from": [str(a), str(b)],
        "n_rows": int(len(big)),
        "n_run_csvs": len(runs_a) + len(runs_b),
    }
    (out / "merge_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    sweep_path = REPO_ROOT / "scripts" / "run_stress_test_sweep.py"
    name = "_pz_merge_stress_sweep"
    spec = importlib.util.spec_from_file_location(name, sweep_path)
    if spec and spec.loader:
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        sig = mod._significance_summary(big)
        sig.to_csv(out / "stress_significance_summary.csv", index=False)
        summ = mod._paired_deltas(big)
        if not summ.empty:
            summ.to_csv(out / "stress_summary_stratified_minus_random.csv", index=False)
            mod._plot_deltas(summ, out / "figures")
        print("Wrote stress_significance_summary.csv and figures/")

    print(f"Wrote combined runs to {out} ({len(big)} rows, {len(runs_a)+len(runs_b)} run CSVs)")


if __name__ == "__main__":
    main()
