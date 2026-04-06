"""
Extract document features from a PDF.

Usage:
    python scripts/extract_features.py path/to/doc.pdf
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


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


@dataclass
class DocumentFeatures:
    path: str
    token_count: int
    section_count: int
    avg_sentence_length: float
    figure_count: int
    table_count: int


def extract_text(pdf_path: str | Path) -> str:
    """Return the full plain-text content of a PDF."""
    reader = PdfReader(str(pdf_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def count_tokens(text: str) -> int:
    """Approximate token count using whitespace splitting (1 word ≈ 1 token)."""
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


def extract_features(pdf_path: str | Path) -> DocumentFeatures:
    """Extract features from a PDF and return a DocumentFeatures instance."""
    text = extract_text(pdf_path)
    return DocumentFeatures(
        path=str(pdf_path),
        token_count=count_tokens(text),
        section_count=count_sections(text),
        avg_sentence_length=avg_sentence_length(text),
        figure_count=count_figures(text),
        table_count=count_tables(text),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract features from a PDF.")
    parser.add_argument("pdf", help="Path to the PDF file.")
    args = parser.parse_args()

    path = Path(args.pdf)
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    features = extract_features(path)
    print(f"File            : {features.path}")
    print(f"Token count     : {features.token_count}")
    print(f"Section count   : {features.section_count}")
    print(f"Avg sent length : {features.avg_sentence_length} words")
    print(f"Figure count    : {features.figure_count}")
    print(f"Table count     : {features.table_count}")


if __name__ == "__main__":
    main()
