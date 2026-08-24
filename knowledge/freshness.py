"""Knowledge freshness controls.

Every corpus document carries an Effective Date header. Freshness rules:

  - fresh:    age <= STALE_DAYS
  - stale:    STALE_DAYS < age <= CRITICAL_DAYS   -> answer carries a warning
  - critical: age > CRITICAL_DAYS                -> answers relying on the doc
                are escalated to a senior underwriter (human-in-the-loop).

The append-only update log (data/update_log.md) supplies the change history
shown on the freshness dashboard and in DocumentInfo.last_change.
"""
from __future__ import annotations

import re
from datetime import date, datetime

STALE_DAYS = 365
CRITICAL_DAYS = 730

_LOG_PATH_DEFAULT = "data/update_log.md"
_LOG_LINE_RE = re.compile(
    r"^\s*[-*\s]*\|?\s*(\d{4}-\d{2}-\d{2})\s*\|\s*([\w\-]+)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|?\s*$"
)


def parse_date(value: str) -> date | None:
    try:
        return datetime.strptime((value or "").strip()[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def age_in_days(effective_date: str, today: date | None = None) -> int:
    d = parse_date(effective_date)
    if d is None:
        return CRITICAL_DAYS + 1  # undated documents are treated as unverified
    today = today or date.today()
    return (today - d).days


def freshness_status(age_days: int) -> str:
    if age_days > CRITICAL_DAYS:
        return "critical"
    if age_days > STALE_DAYS:
        return "stale"
    return "fresh"


def evaluate_freshness(effective_date: str, today: date | None = None) -> tuple[int, str]:
    age = age_in_days(effective_date, today)
    return age, freshness_status(age)


def parse_update_log(base_dir, log_path: str | None = None) -> list[dict]:
    """Parse the append-only update log into structured change records."""
    from pathlib import Path
    base = Path(base_dir)
    path = Path(log_path) if log_path else base / _LOG_PATH_DEFAULT
    changes: list[dict] = []
    if not path.exists():
        return changes
    for raw in path.read_text(encoding="utf-8").splitlines():
        m = _LOG_LINE_RE.match(raw)
        if not m:
            continue
        changes.append({
            "date": m.group(1),
            "document_id": m.group(2),
            "change": m.group(3).strip(),
            "summary": m.group(4).strip(),
            "approved_by": m.group(5).strip(),
        })
    changes.sort(key=lambda c: c["date"], reverse=True)
    return changes


def latest_change_for(changes: list[dict], document_id: str) -> str:
    for c in changes:
        if c["document_id"] == document_id:
            return f"{c['date']}: {c['summary']}"
    return ""


def stale_warning(document_id: str, title: str, version: str,
                  effective_date: str, status: str) -> str:
    if status == "critical":
        return (f"CRITICAL: {title} ({version}) has been in effect since "
                f"{effective_date} and is past its review window; answers citing it "
                f"require senior underwriter verification.")
    return (f"WARNING: {title} ({version}) took effect {effective_date}; verify no newer "
            f"endorsement supersedes it before relying on this answer.")
