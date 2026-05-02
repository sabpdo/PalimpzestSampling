"""
Download a stratified sample of research papers (PDFs) via Semantic Scholar.

Filters for papers with ≥5 citations (2021-2026) that have open-access PDFs,
then randomly samples from that pool.
Strata: 4 domains (CS, Bio/Medical, Physics, Math)
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
        "fraction": 0.25,
        "s2_field": "Computer Science",
    },
    "bio_medical": {
        "fraction": 0.25,
        "s2_field": "Biology",
    },
    "physics": {
        "fraction": 0.25,
        "s2_field": "Physics",
    },
    "math": {
        "fraction": 0.25,
        "s2_field": "Mathematics",
    },
}

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


def download_pdf(url: str, timeout: int = 60) -> tuple[Optional[bytes], Optional[str]]:
    """Download PDF bytes. Returns (bytes, None) on success or (None, reason) on failure."""
    try:
        resp = requests.get(url, timeout=timeout, allow_redirects=True, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        })
        resp.raise_for_status()
        if len(resp.content) < 1000:
            return None, "too_small"
        if not resp.content[:5] == b'%PDF-':
            return None, "not_pdf"
        return resp.content, None
    except requests.exceptions.HTTPError as e:
        reason = f"http_{e.response.status_code}" if e.response else "http_error"
        print(f"  ✗ {reason}")
        return None, reason
    except requests.exceptions.Timeout:
        print(f"  ✗ timeout")
        return None, "timeout"
    except Exception as e:
        print(f"  ✗ {e}")
        return None, "other"


def save_pdf(pdf_bytes: bytes, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pdf_bytes)


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
) -> tuple[list[dict], dict]:
    """
    Sample papers for one domain. Downloads until target_count is reached.
    Returns (metadata, failure_report).
    """
    print(f"\n{'='*60}")
    print(f"Domain: {domain} (target: {target_count} papers)")
    print(f"{'='*60}")

    candidates = fetch_s2_candidates(config["s2_field"], min_citations)

    collected = []
    metadata = []
    failures = {}
    attempts = 0

    for paper in candidates:
        if len(collected) >= target_count:
            break

        attempts += 1
        print(f"  Trying: {paper['title'][:60]}...", end=" ")

        pdf_bytes, fail_reason = download_pdf(paper["pdf_url"])
        if pdf_bytes is None:
            failures[fail_reason] = failures.get(fail_reason, 0) + 1
            continue

        page_count = get_pdf_page_count(pdf_bytes)
        if page_count is None:
            print("✗ couldn't read PDF")
            failures["unreadable_pdf"] = failures.get("unreadable_pdf", 0) + 1
            continue

        # Save
        filename = f"{domain}_{len(collected):03d}.pdf"
        save_path = output_dir / domain / filename
        save_pdf(pdf_bytes, save_path)
        collected.append(paper)

        record = {
            "domain": domain,
            "page_count": page_count,
            "citations": paper["citations"],
            "year": paper["year"],
            "filename": str(save_path.relative_to(output_dir)),
            "id": paper["id"],
            "title": paper["title"],
        }
        metadata.append(record)

        print(f"✓ {page_count}pp, {paper['citations']} cites")
        time.sleep(0.5)

    total_failures = sum(failures.values())
    print(f"  Collected: {len(collected)}")
    print(f"  Attempts: {attempts}  Successes: {len(collected)}  Failures: {total_failures} ({100*total_failures/max(attempts,1):.0f}%)")

    return metadata, {"domain": domain, "attempts": attempts, "successes": len(collected), "failures": failures}


def main():
    parser = argparse.ArgumentParser(description="Download stratified paper sample")
    parser.add_argument("--output-dir", type=str, default="./papers",
                        help="Directory to save PDFs (default: ./papers)")
    parser.add_argument("--total", type=int, default=300,
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
    all_failures = []

    for domain, config in DOMAIN_CONFIG.items():
        target = int(args.total * config["fraction"])
        meta, fail_report = sample_domain(domain, config, target, output_dir, args.min_citations)
        all_metadata.extend(meta)
        all_failures.append(fail_report)

    # Save metadata
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(all_metadata, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    for domain in DOMAIN_CONFIG:
        count = sum(1 for m in all_metadata if m["domain"] == domain)
        print(f"  {domain:15s}  total={count:3d}")
    total = len(all_metadata)
    print(f"  {'TOTAL':15s}  {' '*16}  total={total:3d}")

    # Print failure report
    print(f"\n{'='*60}")
    print("FAILURE REPORT")
    print(f"{'='*60}")
    for report in all_failures:
        domain = report["domain"]
        attempts = report["attempts"]
        successes = report["successes"]
        total_fail = sum(report["failures"].values())
        fail_rate = 100 * total_fail / max(attempts, 1)
        flag = " ← HIGH FAILURE RATE" if fail_rate > 40 else ""
        print(f"  {domain:15s}  {successes}/{attempts} succeeded ({fail_rate:.0f}% failed){flag}")
        for reason, count in sorted(report["failures"].items(), key=lambda x: -x[1]):
            print(f"    {reason:20s}  {count}")

    print(f"\nPDFs saved to: {output_dir.resolve()}")
    print(f"Metadata saved to: {meta_path.resolve()}")


if __name__ == "__main__":
    main()