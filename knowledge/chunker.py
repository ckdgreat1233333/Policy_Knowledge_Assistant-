"""Clause-based chunking for the policy corpus.

Chunking strategy (and why):

Insurance documents are legally structured: every numbered clause is a
self-contained semantic unit that can be cited on its own. Splitting on
clause boundaries therefore:

  1. keeps each chunk topically coherent (no mid-clause truncation),
  2. yields exact, human-verifiable citation refs ("§4 Exclusions"),
  3. lets the agent retrieve *the* clause instead of a window around it,
  4. makes chunk size naturally bounded - clauses rarely exceed ~120 words.

Very long clauses are further split on sentence boundaries so no embedding
is diluted by mixed topics; sub-chunks inherit the parent clause ref.

A corpus document begins with a small metadata header:

    Title: SecurePlus Family Health Policy
    Doc Type: policy
    Product Line: health
    Version: v2.1
    Effective Date: 2026-05-01

followed by numbered clauses ("1.", "3.1", "4.2", ...).
"""
import re

HEADER_FIELD_MAP = {
    "title": "title",
    "doc type": "doc_type",
    "product line": "product_line",
    "policy id": "policy_id",
    "version": "version",
    "effective date": "effective_date",
    "supersedes": "supersedes",
    "approved by": "approved_by",
}

CLAUSE_REF_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\s*[\.\)]?\s+[A-Z]")
CLAUSE_START_RE = re.compile(r"(?:\d+(?:\.\d+)+\.?|[1-9]\d*\.)\s+[A-Z]")
CLAUSE_SPLIT_RE = re.compile(r"(?m)^\s*(?=" + CLAUSE_START_RE.pattern + ")")

MAX_CHUNK_CHARS = 900


def parse_metadata(text: str) -> dict:
    """Parse the leading `Key: value` header block of a corpus document."""
    metadata: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        m = re.match(r"^\s*([A-Za-z][A-Za-z /()&.\-']{2,60}?)\s*:\s*(.*)$", line)
        if not m:
            break
        key = m.group(1).strip().lower()
        value = m.group(2).strip()
        if key in HEADER_FIELD_MAP:
            metadata[HEADER_FIELD_MAP[key]] = value
    return metadata


def split_clauses(text: str) -> list[str]:
    """Split document body into clause strings starting at numbered headers."""
    # Drop the header block before splitting.
    lines = text.splitlines()
    body_start = 0
    for i, raw in enumerate(lines):
        line = raw.rstrip()
        if not line.strip():
            body_start = i + 1
            continue
        if re.match(r"^\s*[A-Za-z][A-Za-z /()&.\-']{2,60}?\s*:", line):
            body_start = i + 1
            continue
        break
    body = "\n".join(lines[body_start:])

    parts = CLAUSE_SPLIT_RE.split(body)
    clauses: list[str] = []
    for part in parts:
        chunk = part.strip()
        if chunk and CLAUSE_REF_RE.match(chunk):
            clauses.extend(_split_long(chunk))
    return clauses


def _split_long(clause: str, limit: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split an over-long clause at sentence boundaries, keeping the ref."""
    if len(clause) <= limit:
        return [clause]
    m = CLAUSE_REF_RE.match(clause)
    if m:
        num_end = m.end(1)
        tail = re.match(r"[\.\)]?\s*", clause[num_end:])
        prefix = clause[:num_end + tail.end()]
    else:
        prefix = ""
    sentences = re.split(r"(?<=[.;])\s+", clause[len(prefix):])
    out, buf = [], prefix
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(buf) + len(s) + 1 > limit and len(buf) > len(prefix):
            out.append(buf.strip())
            buf = prefix + s
        else:
            buf = f"{buf} {s}".strip()
    if len(buf) > len(prefix):
        out.append(buf.strip())
    return out or [clause]
