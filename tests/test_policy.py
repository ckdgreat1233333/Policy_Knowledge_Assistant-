"""Tests for the Policy & Knowledge Management Assistant.

LLM-dependent paths use a deterministic FakeLLM so tests never call the
network; grounding/retrieval/freshness are exercised against the real corpus.
"""
import json
from datetime import date

import pytest

from guardrails.compliance import (
    CUSTOMER_DISCLAIMERS,
    detect_scope_violation,
    enforce_safe_language,
)
from knowledge.chunker import parse_metadata, split_clauses
from knowledge.comparator import PolicyComparator
from knowledge.freshness import (
    CRITICAL_DAYS,
    STALE_DAYS,
    age_in_days,
    evaluate_freshness,
    freshness_status,
    parse_update_log,
)
from knowledge.policy_store import PolicyKnowledgeStore
from agents.policy_agent import PolicyReActAgent, classify_intent


# ─── Fixtures ──────────────────────────────────────────────

class FakeLLM:
    """Returns canned JSON per prompt type; records calls."""

    def __init__(self):
        self.calls = []

    def generate(self, prompt: str, temperature: float = 0.2,
                 max_tokens: int = 512) -> str:
        self.calls.append(prompt)
        if "REACT" in prompt.upper() or "ACTIONS you may take" in prompt:
            return json.dumps({
                "thought": "Search for supporting clauses.",
                "action": "search_clauses",
                "action_input": {"query": "maternity waiting period health"},
            })
        if "internal Policy Assistant" in prompt:
            return json.dumps({
                "answer": ("Maternity is covered after a 24-month wait "
                           "[pol_health_secureplus §3.2]."),
                "key_points": ["24 month waiting [pol_health_secureplus §3.2]"],
                "ambiguities": [],
                "cannot_determine": False,
            })
        if "customer-facing" in prompt:
            return json.dumps({
                "answer": ("Yes - maternity costs are covered once you have held the "
                           "policy for 24 months in a row."),
                "simplified_points": ["Wait 24 months of continuous cover"],
                "cannot_determine": False,
            })
        return "{}"


@pytest.fixture(scope="module")
def store():
    return PolicyKnowledgeStore()


@pytest.fixture()
def agent(store):
    llm = FakeLLM()
    prompts = _Prompts()
    return PolicyReActAgent(store, llm, prompts)


class _Prompts:
    """Loads real templates from prompts/ without an LLM dependency."""

    def load(self, name, **kw):
        from pathlib import Path
        p = Path(__file__).resolve().parent.parent / "prompts" / name
        text = p.read_text(encoding="utf-8")
        try:
            return text.format(**kw)
        except (KeyError, IndexError):
            # templates contain literal JSON braces; escape and retry
            import re
            safe = re.sub(r"\{(\w+)\}", r"%(\1)s", text)
            return safe % kw if kw else text.replace("{", "{{").replace("}", "}}")


# ─── Chunking ──────────────────────────────────────────────

def test_parse_metadata_extracts_header_fields():
    doc = ("Title: Test Policy\nDoc Type: policy\nProduct Line: motor\n"
           "Version: v9.9\nEffective Date: 2026-01-01\n\n1. Scope\nBody here.")
    meta = parse_metadata(doc)
    assert meta["title"] == "Test Policy"
    assert meta["doc_type"] == "policy"
    assert meta["product_line"] == "motor"
    assert meta["version"] == "v9.9"
    assert meta["effective_date"] == "2026-01-01"


def test_split_clauses_yields_citable_chunks():
    doc = ("Title: T\n\n1. First Clause\nContent one.\n\n"
           "2. Second Clause\nContent two.\n3.1 Sub Clause\nSub content.")
    clauses = split_clauses(doc)
    assert len(clauses) == 3
    assert clauses[0].startswith("1. First")
    assert clauses[2].startswith("3.1")


def test_long_clause_is_sentence_split():
    clause = "1. Long Clause " + " ".join(["sentence text here."] * 120)
    parts = split_clauses(clause)
    assert len(parts) >= 2
    assert all(p.startswith("1. ") for p in parts)


# ─── Freshness ─────────────────────────────────────────────

def test_freshness_thresholds():
    today = date(2026, 8, 23)
    fresh_age, fresh_status = evaluate_freshness("2026-05-01", today)
    assert fresh_status == "fresh"
    stale_age, stale_status = evaluate_freshness(
        (date(2026, 8, 23).replace(year=2025, day=1)).strftime("%Y-%m-%d"), today)
    assert stale_status in ("stale", "critical")
    critical_age, critical_status = evaluate_freshness("2024-01-01", today)
    assert critical_status == "critical"


def test_undated_document_treated_as_critical():
    assert freshness_status(age_in_days("")) == "critical"


def test_update_log_parsed():
    from pathlib import Path
    base = Path(__file__).resolve().parent.parent
    changes = parse_update_log(base)
    assert changes, "update log should contain entries"
    entry = changes[0]
    assert {"date", "document_id", "summary", "approved_by"} <= set(entry)


# ─── Store / RAG ───────────────────────────────────────────

def test_corpus_ingested_with_provenance(store):
    assert len(store.chunks) > 20
    doc_ids = {c.document_id for c in store.chunks}
    assert {"pol_health_secureplus", "pol_motor_drivecare",
            "pol_lifeshield_term"} <= doc_ids
    for c in store.chunks[:10]:
        assert c.version and c.effective_date and c.clause_ref


def test_semantic_retrieval_grounds_on_relevant_clause(store):
    hits = store.retrieve("maternity benefit waiting period newborn cover", k=4)
    assert hits
    assert hits[0].similarity >= 0.30
    top_ids = {h.chunk.document_id for h in hits}
    assert "pol_health_secureplus" in top_ids or \
        "end_health_maternity" in top_ids


def test_similarity_threshold_filters_noise(store):
    hits = store.retrieve("quantum chromodynamics particle accelerator", k=4)
    assert hits == []


def test_documents_report_freshness(store):
    infos = {d.document_id: d for d in store.documents()}
    assert infos["pol_lifeshield_term"].freshness_status == "critical"
    assert infos["pol_health_secureplus"].age_days < STALE_DAYS


# ─── Compliance guardrails ─────────────────────────────────

def test_safe_language_scrubs_guarantee():
    dirty = "Good news! Your claim will be approved and this is legally binding."
    clean, violations = enforce_safe_language(dirty)
    assert clean != dirty
    assert violations


def test_scope_refusals():
    kind, msg = detect_scope_violation("Should I sue the insurer in court?")
    assert kind.startswith("refuse")
    kind2, _ = detect_scope_violation("Please approve my claim now.")
    assert kind2.startswith("refuse")
    assert detect_scope_violation("What does my policy exclude?") is None


# ─── Agent ─────────────────────────────────────────────────

def test_intent_classification():
    assert classify_intent("what is not covered under motor?") == "exclusion"
    assert classify_intent("how do I renew after a lapse?") == "renewal"
    assert classify_intent("did the endorsement change zero dep?") == "endorsement"


def test_agent_run_returns_trace_and_validated_citations(agent):
    result = agent.run("Is maternity covered and after what waiting period?",
                       track="business")
    assert result["trace"], "ReAct trace must not be empty"
    assert result["evidence"]
    refs = {(c.document_id, c.clause_ref) for c in result["citations"]}
    ev_refs = {(r.chunk.document_id, r.chunk.clause_ref)
               for r in result["evidence"]}
    assert refs <= ev_refs, "citations must come only from retrieved evidence"


def test_agent_confidence_penalised_for_stale_docs(agent, monkeypatch):
    result = agent.run("Is suicide within 12 months covered under LifeShield?",
                       track="business")
    life_hits = [rc for rc in result["evidence"]
                 if rc.chunk.document_id == "pol_lifeshield_term"]
    if any(rc.freshness_status == "critical" for rc in life_hits):
        assert result["decision"] == "escalate"


# ─── Service orchestration (integration, FakeLLM) ──────────

@pytest.fixture()
def service():
    import services.policy_service as ps
    svc = ps.PolicyService.__new__(ps.PolicyService)
    svc.store = PolicyKnowledgeStore()
    fake = FakeLLM()
    svc.llm = fake
    svc.prompts = _Prompts()
    svc.agent = PolicyReActAgent(svc.store, fake, svc.prompts)
    svc.comparator = PolicyComparator(svc.store, fake, svc.prompts)
    return svc


def test_business_query_structured_response(service):
    ans = service.business_query(
        "Is maternity covered under SecurePlus and what is the waiting period?")
    d = ans.to_dict()
    assert d["decision"] in ("answer", "escalate")
    assert d["reasoning_trace"]
    assert d["disclaimers"]
    for c in d["citations"]:
        assert c["document_id"] and c["clause_ref"] and c["version"]


def test_business_query_refuses_legal_question(service):
    d = service.business_query("Should I go to court over this claim dispute?").to_dict()
    assert d["decision"] == "refuse"


def test_customer_chat_hides_reasoning_and_adds_disclaimers(service):
    d = service.customer_chat("What happens if I miss my renewal date?")
    if hasattr(d, "to_dict"):
        d = d.to_dict()
    assert d["reasoning_trace"] == []
    assert d["retrieved_clauses"] == []
    assert d["disclaimers"] == CUSTOMER_DISCLAIMERS
    assert d["narrative"]


def test_chat_detects_comparison_request(service):
    d = service.customer_chat(
        "Compare SecurePlus Health and DriveCare Motor")
    assert d["type"] == "comparison"
    assert d["decision"] == "answer"
    assert len(d["rows"]) > 5
    assert d["disclaimers"]


def test_chat_normal_question_not_mistaken_for_comparison(service):
    d = service.customer_chat("Is dental treatment covered?")
    assert d.get("type") != "comparison"


def test_comparison_rows_and_disclosures(service):
    out = service.compare_policies("SecurePlus Health", "DriveCare Motor")
    assert out["rows"]
    assert out["disclosures"]
    relations = {row["relation"] for row in out["rows"]}
    assert relations <= {"same", "different", "only_a", "only_b"}
