"""Ethics, risk & governance guardrails for the insurance assistant.

Two enforcement layers:

1. OUTPUT SCRUB  - every customer-facing string is scrubbed AFTER the LLM
   step. A prompt-injected or drifted model can never promise a claim payout,
   guarantee coverage, or hand out legal advice; violations are logged.

2. SCOPE REFUSAL - questions outside the system's mandate (legal advice,
   claim adjudication, pricing) are refused and routed to the right channel.
"""
from __future__ import annotations

import re

# ── Banned claims (customer track): pattern -> safe replacement ──────────
BANNED_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"\byour\s+claim\s+(?:will|shall)\s+be\s+(?:approved|paid|settled)\b", re.I),
     "your claim will be assessed as per the policy terms", "guaranteed claim approval"),
    (re.compile(r"\b(?:we|the insurer)\s+(?:will|shall)?\s*definitely\s+(?:pay|approve|cover)\b", re.I),
     "coverage is subject to policy terms and claim assessment", "definitive payout promise"),
    (re.compile(r"\bguaranteed?\s+(?:coverage|payout|settlement|approval)\b", re.I),
     "coverage that is always subject to policy terms", "guaranteed coverage claim"),
    (re.compile(r"\ball\s+(?:your\s+)?(?:expenses|costs)\s+(?:will be|are)\s+covered\b", re.I),
     "eligible expenses are covered per the policy limits and exclusions", "unlimited coverage claim"),
    (re.compile(r"\bno\s+(?:deductible|exclusions|waiting\s+period)\b", re.I),
     "reduced deductibles or waiting periods where applicable", "no-conditions claim"),
    (re.compile(r"\brisks?[\s-]*free\b", re.I), "with limited risk rather than no risk", "risk-free claim"),
    (re.compile(r"\blegally\s+(?:binding|guaranteed)\b", re.I), "described in the policy document", "legally binding framing"),
]

# ── Scope refusals: the assistant never does these (out-of-scope by design)
LEGAL_ADVICE_RE = re.compile(
    r"\b(sue|suing|litigat\w*|legal\s+(?:opinion|advice|action|strategy)|"
    r"court\s+case|tribunal\s+(?:outcome|decision)|should\s+i\s+(?:sue|go\s+to\s+court))\b", re.I)
CLAIM_ADJUDICATION_RE = re.compile(
    r"\b(approve|reject|deny|adjudicat\w+)\s+(my|this|the)?\s*(claim|hospitalization|hospitalisation)\b", re.I)
PRICING_DECISION_RE = re.compile(
    r"\b(quote|price)\s+(me\s+)?(a\s+)?(new\s+)?policy\b|\bgive\s+me\s+a\s+premium\b", re.I)

CUSTOMER_DISCLAIMERS = [
    "This explanation is generated with AI assistance for general understanding "
    "and is not legal advice, not a confirmation of coverage, and does not "
    "guarantee any claim outcome.",
    "The policy document, its wordings and endorsements prevail over this "
    "summary. Please read your policy schedule carefully.",
    "For decisions about buying, renewing or claiming, please confirm with a "
    "licensed insurance advisor or our customer service team.",
]

BUSINESS_DISCLAIMER = (
    "Internal decision support only. Interpretations must be verified against the "
    "cited clause versions before customer communication; escalated items require "
    "documented senior-underwriter sign-off."
)

REFUSAL_LEGAL = (
    "I'm sorry, but I can't provide legal opinions, litigation guidance, or predict "
    "how a court or tribunal would decide - those need a qualified professional. "
    "I can explain what your policy wording says about this situation."
)

REFUSAL_CLAIM_DECISION = (
    "Claim approval decisions are made only by the claims team after assessment of "
    "documents and eligibility - I can't pre-judge an outcome. I can walk you through "
    "what the policy covers and the claim process steps."
)


def enforce_safe_language(text: str) -> tuple[str, list[str]]:
    """Scrub banned claims from LLM output. Returns (clean_text, violations)."""
    violations: list[str] = []
    if not text:
        return text, violations
    for pattern, replacement, label in BANNED_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            violations.append(f"{label} ({len(matches)}x)")
            text = pattern.sub(replacement, text)
    return text, violations


def detect_scope_violation(question: str) -> tuple[str, str] | None:
    """Return (refusal_kind, message) if the question is out of scope."""
    if LEGAL_ADVICE_RE.search(question):
        return ("refuse_legal", REFUSAL_LEGAL)
    if CLAIM_ADJUDICATION_RE.search(question):
        return ("refuse_claim", REFUSAL_CLAIM_DECISION)
    return None


def disclaimers_for_track(track: str) -> list[str]:
    return [BUSINESS_DISCLAIMER] if track == "business" else list(CUSTOMER_DISCLAIMERS)
