"""
Download a stratified sample of research papers (PDFs) via Semantic Scholar.

Filters for papers with ≥5 citations (2021-2026) that have open-access PDFs,
then randomly samples from that pool.
Strata: 4 domains (CS, Bio/Medical, Physics, Math) x 2 lengths (short, long)
Target: ~200 papers total

Usage:
    pip install requests PyPDF2
    python download_papers.py --output-dir ./papers --total 200
"""

import requests
import random
import json
import argparse
import time
from pathlib import Path
from PyPDF2 import PdfReader
from io import BytesIO
from typing import Optional

# ── Configuration ──────────────────────────────────────────────────────────────

DOMAIN_CONFIG = {
    "cs": {
        "fraction": 0.30,
        "s2_field": "Computer Science",
    },
    "bio": {
        "fraction": 0.30,
        "s2_field": "Biology",
    },
    "physics": {
        "fraction": 0.20,
        "s2_field": "Physics",
    },
    "math": {
        "fraction": 0.20,
        "s2_field": "Mathematics",
    },
}

SHORT_MAX_PAGES = 10
LONG_MIN_PAGES = 11
MIN_CITATIONS = 1
YEAR_RANGE = "2021-2026"

S2_BULK_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"

# ── Helpers ────────────────────────────────────────────────────────────────────

def get_pdf_page_count(pdf_bytes: bytes) -> Optional[int]:
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        return len(reader.pages)
    except Exception:
        return None


def download_pdf(url: str, timeout: int = 60) -> Optional[bytes]:
    try:
        resp = requests.get(url, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()
        if len(resp.content) < 1000:
            return None
        if not resp.content[:5] == b'%PDF-':
            return None
        return resp.content
    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        return None


def save_pdf(pdf_bytes: bytes, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pdf_bytes)


def classify_length(page_count: int) -> Optional[str]:
    if page_count <= SHORT_MAX_PAGES:
        return "short"
    elif page_count >= LONG_MIN_PAGES:
        return "long"
    return None


# ── Semantic Scholar fetching ──────────────────────────────────────────────────

def fetch_s2_candidates(field: str, min_citations: int, pool_size: int = 800) -> list[dict]:
    """
    Fetch a large pool of candidate papers from Semantic Scholar,
    filtered to those with open-access PDFs and ≥min_citations.
    Returns a shuffled list for random sampling.
    """
    candidates = []
    fields = "title,openAccessPdf,citationCount,year,externalIds"
    token = None

    print(f"  Fetching candidates from Semantic Scholar (field={field})...")

    while len(candidates) < pool_size:
        params = {
            "query": "",
            "fieldsOfStudy": field,
            "year": YEAR_RANGE,
            "minCitationCount": min_citations,
            "openAccessPdf": "",
            "fields": fields,
            "limit": 100,
        }
        if token:
            params["token"] = token

        try:
            resp = requests.get(S2_BULK_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  ✗ S2 API error: {e}")
            break

        papers = data.get("data", [])
        if not papers:
            break

        for paper in papers:
            pdf_info = paper.get("openAccessPdf")
            if not pdf_info or not pdf_info.get("url"):
                continue
            candidates.append({
                "id": paper.get("paperId", ""),
                "title": paper.get("title", ""),
                "pdf_url": pdf_info["url"],
                "citations": paper.get("citationCount", 0),
                "year": paper.get("year"),
                "external_ids": paper.get("externalIds", {}),
            })

        token = data.get("token")
        if not token:
            break

        print(f"    ... {len(candidates)} candidates so far")
        time.sleep(1)  # rate limit: ~1 req/sec without API key

    print(f"  Total candidates with open PDFs: {len(candidates)}")
    random.shuffle(candidates)
    return candidates


# ── Main sampling logic ───────────────────────────────────────────────────────

def sample_domain(
    domain: str,
    config: dict,
    target_count: int,
    output_dir: Path,
    min_citations: int,
) -> list[dict]:
    """
    Sample papers for one domain, stratified 50/50 by short/long.
    Iterates through the shuffled candidate pool, downloads PDFs,
    classifies by page count, and saves until buckets are full.
    """
    print(f"\n{'='*60}")
    print(f"Domain: {domain} (target: {target_count} papers)")
    print(f"{'='*60}")

    candidates = fetch_s2_candidates(config["s2_field"], min_citations)

    target_per_length = target_count // 2
    collected = {"short": [], "long": []}
    metadata = []

    for paper in candidates:
        if (len(collected["short"]) >= target_per_length and
                len(collected["long"]) >= target_per_length):
            break

        print(f"  Trying: {paper['title'][:60]}...", end=" ")

        pdf_bytes = download_pdf(paper["pdf_url"])
        if pdf_bytes is None:
            continue

        page_count = get_pdf_page_count(pdf_bytes)
        if page_count is None:
            print("✗ couldn't read PDF")
            continue

        length_class = classify_length(page_count)
        if length_class is None:
            print(f"✗ {page_count}pp (in gap)")
            continue

        if len(collected[length_class]) >= target_per_length:
            print(f"✗ {length_class} bucket full")
            continue

        # Save
        filename = f"{domain}_{length_class}_{len(collected[length_class]):03d}.pdf"
        save_path = output_dir / domain / length_class / filename
        save_pdf(pdf_bytes, save_path)
        collected[length_class].append(paper)

        record = {
            "domain": domain,
            "length_class": length_class,
            "page_count": page_count,
            "citations": paper["citations"],
            "year": paper["year"],
            "filename": str(save_path.relative_to(output_dir)),
            "id": paper["id"],
            "title": paper["title"],
        }
        metadata.append(record)

        print(f"✓ {page_count}pp, {paper['citations']} cites → {length_class}")
        time.sleep(0.5)

    print(f"  Collected: {len(collected['short'])} short, {len(collected['long'])} long")
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Download stratified paper sample")
    parser.add_argument("--output-dir", type=str, default="./papers",
                        help="Directory to save PDFs (default: ./papers)")
    parser.add_argument("--total", type=int, default=200,
                        help="Total number of papers to download (default: 200)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--min-citations", type=int, default=5,
                        help="Minimum citation count filter (default: 5)")
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metadata = []

    for domain, config in DOMAIN_CONFIG.items():
        target = int(args.total * config["fraction"])
        meta = sample_domain(domain, config, target, output_dir, args.min_citations)
        all_metadata.extend(meta)

    # Save metadata
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(all_metadata, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for domain in DOMAIN_CONFIG:
        short = sum(1 for m in all_metadata if m["domain"] == domain and m["length_class"] == "short")
        long_ = sum(1 for m in all_metadata if m["domain"] == domain and m["length_class"] == "long")
        print(f"  {domain:15s}  short={short:3d}  long={long_:3d}  total={short+long_:3d}")
    total = len(all_metadata)
    print(f"  {'TOTAL':15s}  {' '*16}  total={total:3d}")
    print(f"\nPDFs saved to: {output_dir.resolve()}")
    print(f"Metadata saved to: {meta_path.resolve()}")


if __name__ == "__main__":
    main()