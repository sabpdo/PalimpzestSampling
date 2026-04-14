"""
Extract document features from a PDF.

Usage:
    python scripts/extract_features.py path/to/doc.pdf
    python scripts/extract_features.py --scan ./papers -o papers/paper_features.csv
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
from pypdf import PdfReader
from tqdm import tqdm

# Path to the word frequency list (line number = rank, 1-indexed)
_FREQ_LIST_PATH = Path(__file__).parent / "google-10000-english-no-swears.txt"
_RARE_RANK_THRESHOLD = 10_000
_MAX_RANK = _RARE_RANK_THRESHOLD + 1  # assigned to words not in the list

# Cached rank lookup: word -> 1-based rank
_word_rank: dict[str, int] | None = None


def _load_word_ranks() -> dict[str, int]:
    global _word_rank
    if _word_rank is None:
        _word_rank = {}
        with open(_FREQ_LIST_PATH) as f:
            for rank, line in enumerate(f, start=1):
                word = line.strip().lower()
                if word:
                    _word_rank[word] = rank
    return _word_rank


def _rank(word: str, ranks: dict[str, int]) -> int:
    return ranks.get(word.lower(), _MAX_RANK)


def _count_syllables(word: str) -> int:
    """Approximate syllable count by counting vowel clusters."""
    return len(re.findall(r"[aeiouAEIOU]+", word)) or 1


# Split on sentence-ending punctuation regardless of what follows
_PUNCT_SPLIT_RE = re.compile(r"[.!?]+")


# Numbered heading: "1. Title", "1.1 Title", "2.3.1 Title" — word after number must be title-case
_NUMBERED_SECTION_RE = re.compile(r"^\d+(?:\.\d+)*\.?\s+[A-Z][a-z]")
# Keyword-prefixed: "Chapter 1", "Section 2", "Appendix A"
_KEYWORD_SECTION_RE = re.compile(r"^(?:chapter|section|appendix)\s+[\dA-Z]", re.IGNORECASE)
# ALL-CAPS heading: entire line is uppercase words (no IGNORECASE), short enough to be a header
_ALLCAPS_SECTION_RE = re.compile(r"^[A-Z][A-Z\s\-]{2,}$")

# Sentence boundary: split on ". ", "! ", "? " followed by a capital letter
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
# Figures: "Figure 1", "Fig. 1", "Fig 1" — capture the number for deduplication
_FIGURE_RE = re.compile(r"\bfig(?:ure|\.?)?\s*(\d+)", re.IGNORECASE)
# Tables: "Table 1", "Tab. 1", "Tab 1" — capture the number for deduplication
_TABLE_RE = re.compile(r"\btab(?:le|\.?)?\s*(\d+)", re.IGNORECASE)


# Columns used by ``feature_stratified_sampling`` (must match ``FEATURE_COLUMNS`` there).
STRATIFICATION_FEATURE_FIELDS = (
    "word_count",
    "section_count",
    "avg_sentence_length",
    "figure_count",
    "table_count",
    "complexity_score",
)


@dataclass
class DocumentFeatures:
    path: str
    word_count: int
    section_count: int
    avg_sentence_length: float
    figure_count: int
    table_count: int
    complexity_score: float


def extract_text(pdf_path: str | Path) -> str:
    """Return the full plain-text content of a PDF."""
    reader = PdfReader(str(pdf_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def count_words(text: str) -> int:
    """Approximate word count using whitespace splitting (1 word ≈ 1 word)."""
    return len(text.split())


def count_sections(text: str) -> int:
    """Count likely section headers in the document text."""
    count = 0
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and (
            _NUMBERED_SECTION_RE.match(stripped)
            or _KEYWORD_SECTION_RE.match(stripped)
            or (_ALLCAPS_SECTION_RE.match(stripped) and len(stripped) <= 60)
        ):
            count += 1
    return count


def count_figures(text: str) -> int:
    """Count distinct figures by deduplicating on figure number (e.g. 'Figure 1' and 'Fig. 1' both → 1)."""
    return len(set(_FIGURE_RE.findall(text)))


def count_tables(text: str) -> int:
    """Count distinct tables by deduplicating on table number (e.g. 'Table 1' and 'Tab. 1' both → 1)."""
    return len(set(_TABLE_RE.findall(text)))


def avg_sentence_length(text: str) -> float:
    """Average number of words per sentence."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]
    if not sentences:
        return 0.0
    return round(sum(len(s.split()) for s in sentences) / len(sentences), 2)


def complexity_score(text: str) -> float:
    """Compute a weighted document complexity score in [0, 1].

    Score = 0.35 * word_rarity
          + 0.20 * avg_sentence_length  (normalised)
          + 0.20 * rare_word_density
          + 0.15 * lexical_diversity
          + 0.10 * syllable_load        (normalised)
    """
    ranks = _load_word_ranks()
    words = [t for t in re.findall(r"[a-zA-Z]+", text)]
    if not words:
        return 0.0

    # --- word rarity: mean log-normalised rank ---
    log_max = math.log(_MAX_RANK)
    word_rarity = sum(math.log(_rank(t, ranks)) / log_max for t in words) / len(words)

    # --- average sentence length (split on punctuation), normalised to [0,1] ---
    segments = [s.strip() for s in _PUNCT_SPLIT_RE.split(text) if s.strip()]
    raw_avg_sent_len = (
        sum(len(s.split()) for s in segments) / len(segments) if segments else 0.0
    )
    avg_sent_len_norm = min(raw_avg_sent_len / 40.0, 1.0)

    # --- rare word density: fraction of words with rank > threshold ---
    rare_word_density = sum(1 for t in words if _rank(t, ranks) >= _MAX_RANK) / len(words)

    # --- lexical diversity: type-word ratio ---
    lexical_diversity = len({t.lower() for t in words}) / len(words)

    # --- syllable load: mean syllables per word, normalised to [0,1] ---
    raw_syllable_load = sum(_count_syllables(t) for t in words) / len(words)
    syllable_load_norm = min(raw_syllable_load / 4.0, 1.0)

    score = (
        0.35 * word_rarity
        + 0.20 * avg_sent_len_norm
        + 0.20 * rare_word_density
        + 0.15 * lexical_diversity
        + 0.10 * syllable_load_norm
    )
    return round(score, 4)


def extract_features(pdf_path: str | Path) -> DocumentFeatures:
    """Extract features from a PDF and return a DocumentFeatures instance."""
    text = extract_text(pdf_path)
    return DocumentFeatures(
        path=str(pdf_path),
        word_count=count_words(text),
        section_count=count_sections(text),
        avg_sentence_length=avg_sentence_length(text),
        figure_count=count_figures(text),
        table_count=count_tables(text),
        complexity_score=complexity_score(text),
    )


def paper_domain_from_path(pdf_path: str | Path) -> str:
    """
    Infer paper domain from the folder immediately under ``papers/``.

    Example: ``papers/cs/cs_001.pdf`` -> ``cs``.
    """
    p = Path(pdf_path)
    parts = p.parts
    for i, part in enumerate(parts):
        if part == "papers" and i + 1 < len(parts):
            return parts[i + 1]
    return "unknown"


def iter_pdf_paths(papers_root: Path) -> list[Path]:
    """All ``*.pdf`` under ``papers_root``, sorted for stable row indices."""
    if not papers_root.is_dir():
        return []
    return sorted(papers_root.rglob("*.pdf"))


def build_feature_table(papers_root: Path) -> pd.DataFrame:
    """Extract features for every PDF under ``papers_root``; ``row_index`` matches sort order."""
    paths = iter_pdf_paths(papers_root)
    rows: list[dict] = []
    for row_index, pdf in enumerate(tqdm(paths, desc="PDFs", unit="file")):
        feat = extract_features(pdf)
        d = asdict(feat)
        d["row_index"] = row_index
        d["domain"] = paper_domain_from_path(pdf)
        rows.append(d)
    cols = ["row_index", "path", "domain", *STRATIFICATION_FEATURE_FIELDS]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    return df[cols]


def write_feature_table_csv(papers_root: Path, out_csv: Path) -> pd.DataFrame:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = build_feature_table(papers_root)
    df.to_csv(out_csv, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract features from PDF(s); use --scan to build a stratification table CSV."
    )
    parser.add_argument(
        "pdf",
        nargs="?",
        default=None,
        help="Path to a single PDF (omit if using --scan).",
    )
    parser.add_argument(
        "--scan",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directory of PDFs (e.g. ./papers); writes --output CSV with row_index and features.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("papers/paper_features.csv"),
        help="Output CSV for --scan (default: papers/paper_features.csv).",
    )
    args = parser.parse_args()

    if args.scan is not None:
        root = args.scan
        if not root.is_dir():
            print(f"Error: not a directory: {root}", file=sys.stderr)
            sys.exit(1)
        df = write_feature_table_csv(root, args.output)
        print(f"Wrote {len(df)} rows to {args.output.resolve()}")
        return

    if args.pdf is None:
        parser.error("Provide a PDF path or --scan DIR")
    path = Path(args.pdf)
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    features = extract_features(path)
    print(f"File            : {features.path}")
    print(f"Word count      : {features.word_count}")
    # print(f"Section count   : {features.section_count}")
    print(f"Avg sent length : {features.avg_sentence_length} words")
    print(f"Figure count    : {features.figure_count}")
    print(f"Table count     : {features.table_count}")
    print(f"Complexity score: {features.complexity_score}")


if __name__ == "__main__":
    main()
