"""Policy service facade - wires both interaction tracks.

Business track (underwriters / service teams):
    scope check -> ReAct agent (retrieve -> validate -> decide) ->
    structured answer with citations, trace, confidence, escalation

Customer track:
    same grounded pipeline but with context control (top-4 clauses),
    simplified-language generation and post-generation compliance scrubbing.

Every interaction is written to the audit ledger.
"""
from __future__ import annotations

import logging
import re
import uuid

import database.db as db
from agents.policy_agent import PolicyReActAgent
from guardrails.compliance import (
    CUSTOMER_DISCLAIMERS,
    disclaimers_for_track,
    detect_scope_violation,
    enforce_safe_language,
)
from knowledge.comparator import PolicyComparator
from knowledge.freshness import stale_warning
from knowledge.policy_store import PolicyKnowledgeStore
from models.policy import PolicyAnswer, RetrievedClause
from services.llm_service import LLMService
from services.prompt_service import PromptService

logger = logging.getLogger("policy_service")

CONTEXT_CONTROL_CUSTOMER = 4   # prevent overload: fewer clauses for customers
CONTEXT_CONTROL_BUSINESS = 8

FOLLOWUPS = {
    "exclusion": ["What waiting periods apply to this exclusion?",
                  "Which endorsements modify this clause?"],
    "coverage": ["What are the exclusions for this benefit?",
                 "What is the claim process for this cover?"],
    "endorsement": ["Show me the freshness status of this document.",
                    "What did the previous version say?"],
    "renewal": ["What happens if I miss the renewal date?",
                "Is portability available for this product line?"],
    "claim_process": ["What documents are required for cashless?",
                      "What are the claim decision timelines?"],
    "eligibility": ["What are the entry age limits?",
                    "What waiting periods apply at enrollment?"],
    "general": ["Which documents govern this area?",
                "Has any endorsement changed this recently?"],
}


class PolicyService:

    def __init__(self):
        self.store = PolicyKnowledgeStore()
        self.llm = LLMService()
        self.prompts = PromptService()
        self.agent = PolicyReActAgent(self.store, self.llm, self.prompts)
        self.comparator = PolicyComparator(self.store, self.llm, self.prompts)

    # ── Business track ─────────────────────────────────────────

    def business_query(self, question: str) -> PolicyAnswer:
        refusal = detect_scope_violation(question)
        if refusal:
            return self._refusal("business", question, *refusal)

        raw = self.agent.run(question, track="business")
        narrative, violations = enforce_safe_language(
            str(raw["generated"].get("answer", "")).strip())

        evidence: list[RetrievedClause] = raw["evidence"][:CONTEXT_CONTROL_BUSINESS]
        answer = PolicyAnswer(
            track="business", question=question,
            answered=raw["decision"] == "answer",
            decision=raw["decision"], intent=raw["intent"],
            narrative=narrative or raw.get("escalation_reason", ""),
            confidence=raw["confidence"],
            citations=raw["citations"],
            reasoning_trace=raw["trace"],
            retrieved_clauses=evidence,
            disclaimers=disclaimers_for_track("business"),
            needs_human_override=raw["decision"] != "answer",
            escalation_reason=raw.get("escalation_reason", ""),
            suggested_followups=FOLLOWUPS.get(raw["intent"], FOLLOWUPS["general"]),
        )
        if raw.get("ambiguities"):
            answer.stale_warnings.extend(raw["ambiguities"])
        answer.stale_warnings.extend(self._stale_warnings(evidence))
        if violations:
            answer.compliance_notes.append(
                "non-guarantee scrub applied: " + ", ".join(violations))
        answer.compliance_notes.append(
            f"Grounded on {len(answer.citations)} validated clause citation(s); "
            f"retrieval threshold enforced at {evidence[0].similarity:.2f} top similarity."
            if evidence else "No retrieval grounding available.")
        self._log(answer)
        return answer

    # ── Customer track ─────────────────────────────────────────

    COMPARE_TRIGGER_RE = re.compile(
        r"\b(compare|comparison|difference[s]? between|differences between)\b", re.I)
    COMPARE_A_RE = re.compile(
        r"(?:compare|comparison(?:\s+of)?|difference[s]? between)\s+(.+?)\s+"
        r"(?:and|with|to|vs\.?|versus)\s+(.+?)[?.!]*\s*$", re.I | re.S)
    COMPARE_B_RE = re.compile(r"(.+?)\s+(?:vs\.?|versus)\s+(.+?)[?.!]*\s*$",
                              re.I | re.S)

    def _detect_comparison(self, message: str):
        """Return (id_a, id_b) when the message clearly asks to compare two
        known policies, else None (falls through to normal RAG chat)."""
        if not self.COMPARE_TRIGGER_RE.search(message or ""):
            return None
        m = self.COMPARE_A_RE.search(message.strip()) \
            or self.COMPARE_B_RE.search(message.strip())
        if not m:
            return None
        raw_a = re.sub(r"^\s*compare\s+", "", m.group(1), flags=re.I).strip(" \"'")
        raw_b = m.group(2).strip(" \"'")
        doc_a = self.store.find_document(raw_a)
        doc_b = self.store.find_document(raw_b)
        if not doc_a or not doc_b:
            return None
        if {doc_a.document_id, doc_b.document_id} == {doc_a.document_id}:
            return None  # same document resolved twice
        if "policy" not in (doc_a.doc_type, ) and doc_a.doc_type != "policy":
            return None
        if doc_b.doc_type != "policy":
            return None
        return doc_a.document_id, doc_b.document_id

    def customer_chat(self, message: str) -> dict:
        refusal = detect_scope_violation(message)
        if refusal:
            return self._refusal("customer", message, *refusal).to_dict()

        pair = self._detect_comparison(message)
        if pair:
            return self._comparison_answer(message, *pair)

        return self._chat_standard(message)

    def _comparison_answer(self, message: str, id_a: str, id_b: str) -> dict:
        result = self.compare_policies(id_a, id_b)
        result.update({
            "type": "comparison", "decision": "answer", "intent": "comparison",
            "confidence": 0.85, "question": message,
            "disclaimers": disclaimers_for_track("customer"),
            "suggested_followups": [
                f"Tell me more about {result['policy_a']['title']}",
                "What is not covered under my policy?"],
        })
        try:
            db.create_policy_session(
                session_id=f"PKM-{uuid.uuid4().hex[:10]}", track="customer",
                question=message[:200], decision="answer", intent="comparison",
                answered=True, needs_human_override=False,
                escalation_reason="",
                payload=f"rows={len(result['rows'])};{id_a} vs {id_b}")
        except Exception as exc:
            logger.warning("Session log failed: %s", exc)
        return result

    def _chat_standard(self, message: str) -> dict:
        raw = self.agent.run(message, track="customer")
        narrative, violations = enforce_safe_language(
            str(raw["generated"].get("simplified_points") and
                raw["generated"].get("answer") or "").strip())
        points = [enforce_safe_language(p)[0]
                  for p in (raw["generated"].get("simplified_points") or [])[:4]]

        evidence: list[RetrievedClause] = raw["evidence"][:CONTEXT_CONTROL_CUSTOMER]
        if points:
            narrative = narrative + "\n\n" + "\n".join(f"• {p}" for p in points)

        citations = raw["citations"] or [
            _evidence_citation(rc) for rc in evidence[:2]]

        answered = raw["decision"] == "answer" or bool(narrative)
        answer = PolicyAnswer(
            track="customer", question=message,
            answered=bool(answered), decision=raw["decision"],
            intent=raw["intent"], narrative=narrative,
            confidence=raw["confidence"], citations=citations,
            reasoning_trace=[],          # internal reasoning never shown to customers
            retrieved_clauses=[],
            disclaimers=disclaimers_for_track("customer"),
            needs_human_override=False,
            escalation_reason=(raw.get("escalation_reason", "")
                               if raw["decision"] == "escalate" else ""),
            suggested_followups=FOLLOWUPS.get(raw["intent"], FOLLOWUPS["general"]),
        )
        if raw["decision"] == "escalate" and not narrative:
            answer.narrative = (
                "I couldn't find a clear answer in the current policy wordings for "
                "this. Our team will confirm the exact position for your case - "
                "please reach out via the service desk so nothing is missed.")
        if violations:
            answer.compliance_notes.append(
                "safe-language scrub applied: " + ", ".join(violations))
        answer.compliance_notes.append(
            "Simplified summary; policy document prevails over this explanation.")
        self._log(answer)
        return answer.to_dict()

    # ── Comparison ─────────────────────────────────────────────

    def compare_policies(self, id_a: str, id_b: str):
        result = self.comparator.compare(id_a, id_b)
        from guardrails.compliance import CUSTOMER_DISCLAIMERS
        result.disclosures = list(CUSTOMER_DISCLAIMERS) + [
            "Similarity scores are semantic alignment estimates between clause "
            "texts, not legal equivalence.",
            "This comparison omits premium, pricing and eligibility details; "
            "request an official benefit illustration before deciding."]
        return result.to_dict()

    # ── Knowledge dashboard ────────────────────────────────────

    def freshness_dashboard(self) -> dict:
        docs = [d.to_dict() for d in self.store.documents()]
        return {"documents": docs,
                "counts": {
                    "fresh": sum(1 for d in docs if d["freshness_status"] == "fresh"),
                    "stale": sum(1 for d in docs if d["freshness_status"] == "stale"),
                    "critical": sum(1 for d in docs if d["freshness_status"] == "critical"),
                }}

    # ── Helpers ────────────────────────────────────────────────

    @staticmethod
    def _stale_warnings(evidence: list[RetrievedClause]) -> list[str]:
        seen, out = set(), []
        for rc in evidence:
            if rc.freshness_status == "fresh" or rc.chunk.document_id in seen:
                continue
            seen.add(rc.chunk.document_id)
            out.append(stale_warning(rc.chunk.document_id, rc.chunk.title,
                                     rc.chunk.version, rc.chunk.effective_date,
                                     rc.freshness_status))
        return out

    def _refusal(self, track: str, question: str,
                 kind: str, message: str) -> PolicyAnswer:
        answer = PolicyAnswer(
            track=track, question=question, answered=True, decision="refuse",
            intent="out_of_scope", narrative=message, confidence=0.99,
            disclaimers=disclaimers_for_track(track),
            compliance_notes=["out-of-scope request refused by scope guardrail "
                              f"({kind})"])
        self._log(answer)
        return answer

    def _log(self, answer: PolicyAnswer) -> None:
        try:
            db.create_policy_session(
                session_id=f"PKM-{uuid.uuid4().hex[:10]}",
                track=answer.track, question=answer.question[:200],
                decision=answer.decision, intent=answer.intent,
                answered=answer.answered,
                needs_human_override=answer.needs_human_override,
                escalation_reason=answer.escalation_reason,
                payload=f"conf={answer.confidence:.2f};"
                        f"cites={len(answer.citations)}")
        except Exception as exc:
            logger.warning("Session log failed: %s", exc)


def _evidence_citation(rc: RetrievedClause):
    """Source-hint citation for customer answers (top retrieved clause)."""
    from models.policy import Citation
    return Citation(
        document_id=rc.chunk.document_id, title=rc.chunk.title,
        clause_ref=rc.chunk.clause_ref, version=rc.chunk.version,
        effective_date=rc.chunk.effective_date, similarity=rc.similarity)


_default: PolicyService | None = None


def get_policy_service() -> PolicyService:
    global _default
    if _default is None:
        _default = PolicyService()
    return _default
