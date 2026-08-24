"""Custom ReAct agent for policy reasoning.

Every turn follows the classic loop:

    Thought     : interpret query intent + what evidence is missing
    Action      : retrieve relevant clauses (search_clauses | inspect_document)
    Observation : validate completeness - similarity scores, clause coverage,
                  version/freshness of every retrieved document
    Decision    : answer from sources OR escalate to a senior underwriter

Escalation is a first-class outcome triggered deterministically when evidence
is absent, cited documents are critically stale, confidence is low, or the
grounded generator reports it cannot determine an answer.

If any LLM call fails or returns unparseable output the agent degrades to a
deterministic retrieval+template path so answers stay source-based.
"""
from __future__ import annotations

import json
import logging
import re

from models.policy import AgentStep, Citation, RetrievedClause

logger = logging.getLogger("policy_react_agent")

MAX_STEPS = 4
MIN_CONFIDENCE_TO_ANSWER = 0.45
CITE_RE = re.compile(r"([A-Za-z][\w\-]{3,})\s*§\s*(\d+(?:\.\d+)*)")

INTENT_RULES: list[tuple[str, re.Pattern]] = [
    ("endorsement", re.compile(r"\b(endorsement|revised|revision|supersede|"
                               r"new version|updated policy|changed)\b", re.I)),
    ("renewal", re.compile(r"\b(renew\w*|grace period|lapse[d]?|portab\w*|ncb|"
                           r"no claim bonus|continuity)\b", re.I)),
    ("claim_process", re.compile(r"\b(how (do|to) (i )?claim|claim process|documents? "
                                 r"(needed|required)|cashless|reimbursement|intimation)\b", re.I)),
    ("eligibility", re.compile(r"\b(eligib\w*|entry age|who can|can i buy|min(?:imum)? age|"
                               r"max(?:imum)? age)\b", re.I)),
    ("exclusion", re.compile(r"\b(not\s+cover\w*|exclusions?|excluded|not\s+payable|"
                             r"won'?t\s+cover|limitation)\b", re.I)),
    ("coverage", re.compile(r"\b(covere?d?|coverage|cover\b|benefit|payout|sum insured|"
                            r"pay for|waiting period|maternity)\b", re.I)),
]


def classify_intent(question: str) -> str:
    for intent, pattern in INTENT_RULES:
        if pattern.search(question):
            return intent
    return "general"


class PolicyReActAgent:

    def __init__(self, store, llm, prompts):
        self.store = store
        self.llm = llm
        self.prompts = prompts

    def run(self, question: str, track: str) -> dict:
        """Execute the ReAct loop. Returns a raw result dict."""
        trace: list[AgentStep] = []
        evidence: dict[str, RetrievedClause] = {}
        intent = classify_intent(question)
        empty_rounds = 0

        for step_no in range(1, MAX_STEPS + 1):
            thought, action, action_input = self._next_step(
                question, intent, trace, step_no)

            if action == "finish":
                trace.append(AgentStep(
                    step_no, thought, "finish", "",
                    f"Evidence complete: {len(evidence)} distinct clauses gathered."))
                break

            observation, hits = self._execute_action(action, action_input,
                                                     question, evidence)
            trace.append(AgentStep(step_no, thought, action, action_input, observation))
            empty_rounds = empty_rounds + 1 if hits == 0 else 0
            if empty_rounds >= 2:
                break
            best = max((r.similarity for r in evidence.values()), default=0.0)
            if step_no >= 2 and len(evidence) >= 5 and best >= 0.60:
                trace.append(AgentStep(
                    step_no + 1,
                    f"Evidence is sufficient: {len(evidence)} relevant clauses "
                    f"with top similarity {best:.2f}.",
                    "finish", "",
                    f"Proceeding to grounded generation with {len(evidence)} clauses."))
                break

        context = self._context_block(evidence)
        generated = self._generate_answer(question, track, context)

        result = {
            "intent": intent,
            "trace": trace,
            "evidence": sorted(evidence.values(),
                               key=lambda r: r.similarity, reverse=True),
            "generated": generated,
            "citations": self._validate_citations(generated.get("answer", ""),
                                                  evidence),
            "cannot_determine": bool(generated.get("cannot_determine")),
            "ambiguities": list(generated.get("ambiguities")
                                or generated.get("limitations") or []),
        }
        result["confidence"] = self._score_confidence(result)
        result["decision"], result["escalation_reason"] = self._decide(result)
        return result

    # ── ReAct steps ────────────────────────────────────────────

    def _next_step(self, question: str, intent: str,
                   trace: list[AgentStep], step_no: int):
        if not trace:
            return (f"The user asks about '{question}'. Intent looks like '{intent}'. "
                    f"I need supporting clauses, so I will run a semantic search "
                    f"with insurance-precise terminology."), "search_clauses", question

        trace_text = "\n".join(
            f"Thought: {s.thought}\nAction: {s.action} {s.action_input}\n"
            f"Observation: {s.observation[:600]}" for s in trace)
        prompt = self.prompts.load("react_step_prompt.txt", question=question,
                                   intent=intent, trace=trace_text)
        raw = self._safe_generate(prompt, temperature=0.1, max_tokens=400)
        parsed = _extract_json(raw) or {}
        thought = str(parsed.get("thought") or
                      f"Reviewing observations for '{question}'.")[:500]
        action = str(parsed.get("action") or "finish").strip()
        if action not in ("search_clauses", "inspect_document", "finish"):
            action = "finish"
        action_input = ""
        ai = parsed.get("action_input")
        if isinstance(ai, dict):
            action_input = str(ai.get("query") or ai.get("document_id") or "")
        elif isinstance(ai, str):
            action_input = ai
        if action == "search_clauses" and not action_input:
            action_input = question
        return thought, action, action_input

    def _execute_action(self, action: str, action_input: str, question: str,
                        evidence: dict) -> tuple[str, int]:
        if action == "inspect_document" and action_input:
            chunk = self.store.find_document(action_input)
            if not chunk:
                return f"No document matches '{action_input}'.", 0
            clauses = self.store.get_document_chunks(chunk.document_id)
            for rc in clauses:
                evidence.setdefault(rc.chunk.chunk_id, rc)
            listed = "; ".join(f"§{c.chunk.clause_ref}" for c in clauses[:12])
            return (f"{chunk.title} ({chunk.version}): {len(clauses)} clauses - {listed}",
                    len(clauses))

        query = action_input or question
        hits = self.store.retrieve(query, k=6)
        new = [h for h in hits if h.chunk.chunk_id not in evidence]
        for rc in hits:
            evidence.setdefault(rc.chunk.chunk_id, rc)
        if not hits:
            return (f"No clause exceeded the similarity threshold for '{query}'. "
                    f"Query may need broader phrasing."), 0
        lines = [f"{r.ref} sim={r.similarity:.2f}: {r.citation_text(160)}"
                 for r in hits[:6]]
        return (f"{len(hits)} clauses retrieved ({len(new)} new):\n" +
                "\n".join(lines)), len(hits)

    # ── Grounded generation ────────────────────────────────────

    def _generate_answer(self, question: str, track: str, context: str) -> dict:
        if not context:
            return {"answer": "", "cannot_determine": True,
                    "ambiguities": ["No policy clause passed retrieval thresholds."]}
        template = ("business_answer_prompt.txt" if track == "business"
                    else "customer_answer_prompt.txt")
        prompt = self.prompts.load(template, question=question, context=context)
        raw = self._safe_generate(prompt, temperature=0.2, max_tokens=1400)
        parsed = _extract_json(raw)
        if parsed and isinstance(parsed.get("answer"), str) and parsed["answer"].strip():
            return parsed
        salvaged = _salvage_answer_field(raw) if raw else ""
        if salvaged:
            return {"answer": salvaged, "key_points": [],
                    "cannot_determine": False, "_degraded": False,
                    "_salvaged": True}
        return self._fallback_answer(track, context, question)

    def _fallback_answer(self, track: str, context: str, question: str) -> dict:
        top = [line.split(": ", 1)[0] for line in context.splitlines()[:3]]
        refs = "; ".join(top)
        answer = (
            f"Regarding \"{question}\", the most relevant source material found is: {refs}. "
            f"Please review these clauses directly or route this query through the "
            f"standard verification workflow."
            if track == "business" else
            f"We found some relevant policy information for your question. The key "
            f"points come from: {refs}. Our service team can walk you through the "
            f"details before you make any decision."
        )
        return {"answer": answer, "key_points": [], "cannot_determine": False,
                "_degraded": True}

    # ── Validation / scoring ───────────────────────────────────

    def _validate_citations(self, answer: str,
                            evidence: dict) -> list[Citation]:
        citations: list[Citation] = []
        seen = set()
        by_doc: dict[str, list[RetrievedClause]] = {}
        for rc in evidence.values():
            c = rc.chunk
            keys = {c.document_id.lower()}
            if c.policy_id:
                keys.add(c.policy_id.lower())
            for word in c.title.lower().split():
                if len(word) > 3:
                    keys.add(word)
            for k in keys:
                by_doc.setdefault(k, []).append(rc)
        for doc_token, ref in CITE_RE.findall(answer or ""):
            key = f"{doc_token}-{ref.replace('.', '_')}"
            rc = evidence.get(key)
            if rc is None:
                tl = doc_token.lower()
                pool = by_doc.get(tl, [])
                if not pool:  # partial doc-id match e.g. "secureplus"
                    pool = [v for k, vs in by_doc.items()
                            if tl in k for v in vs]
                exact = [c for c in pool if c.chunk.clause_ref == ref]
                prefix = [c for c in pool if c.chunk.clause_ref.startswith(ref)]
                rc = (exact or prefix or [None])[0]
            if rc is None:
                continue
            ckey = f"{rc.chunk.document_id}-{rc.chunk.clause_ref}"
            if ckey in seen:
                continue
            seen.add(ckey)
            citations.append(Citation(
                document_id=rc.chunk.document_id, title=rc.chunk.title,
                clause_ref=rc.chunk.clause_ref, version=rc.chunk.version,
                effective_date=rc.chunk.effective_date,
                similarity=rc.similarity))
        return citations

    def _score_confidence(self, result: dict) -> float:
        evidence: list[RetrievedClause] = result["evidence"]
        if not evidence:
            return 0.05
        top_sims = [r.similarity for r in evidence[:3]]
        base = sum(top_sims) / len(top_sims)
        if len(evidence) < 2:
            base *= 0.85
        cited_keys = {f"{c.document_id}-{c.clause_ref.replace('.', '_')}"
                      for c in result["citations"]}
        statuses = {rc.freshness_status for rc in evidence
                    if rc in evidence[:3] or rc.chunk.chunk_id in cited_keys}
        if "critical" in statuses:
            base *= 0.6
        elif "stale" in statuses:
            base *= 0.8
        if result.get("cannot_determine"):
            base *= 0.4
        if result["generated"].get("_degraded"):
            base *= 0.75
        return max(0.02, min(0.97, base))

    def _decide(self, result: dict) -> tuple[str, str]:
        if result["generated"].get("_degraded"):
            return "escalate", ("Generator fell back to template mode; answer needs "
                                "human verification.")
        if not result["evidence"]:
            return "escalate", "No policy clause passed the relevance threshold."
        if result["cannot_determine"]:
            return "escalate", ("Grounded generation could not determine the answer "
                                "from retrieved clauses.")
        critical = sorted({rc.chunk.title for rc in result["evidence"][:5]
                           if rc.freshness_status == "critical"})
        if critical:
            return ("escalate",
                    f"Cited document(s) critically stale: {', '.join(critical)}. "
                    "Senior underwriter must verify against current wordings.")
        if result["confidence"] < MIN_CONFIDENCE_TO_ANSWER:
            return ("escalate",
                    f"Confidence {result['confidence']:.2f} below threshold "
                    f"{MIN_CONFIDENCE_TO_ANSWER}; ambiguous or incomplete evidence.")
        return "answer", ""

    # ── Helpers ────────────────────────────────────────────────

    def _context_block(self, evidence: dict, limit: int = 8) -> str:
        rows = sorted(evidence.values(), key=lambda r: r.similarity,
                      reverse=True)[:limit]
        return "\n\n".join(f"[{r.ref}]\n{r.chunk.text}" for r in rows)

    def _safe_generate(self, prompt: str, temperature: float,
                       max_tokens: int) -> str:
        try:
            return self.llm.generate(prompt, temperature=temperature,
                                     max_tokens=max_tokens)
        except Exception as exc:
            logger.warning("LLM call failed: %s", exc)
            return ""


def _extract_json(raw: str) -> dict | None:
    if not raw:
        return None
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(cleaned[start:end + 1])
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        # Truncated JSON: salvage the "answer" field if it is complete.
        return None


_ANSWER_FIELD_RE = re.compile(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"')


def _salvage_answer_field(raw: str) -> str:
    """Recover the answer string from truncated/invalid JSON output."""
    m = _ANSWER_FIELD_RE.search(raw or "")
    if not m:
        return ""
    try:
        return json.loads(f'"{m.group(1)}"').strip()
    except json.JSONDecodeError:
        return m.group(1).replace('\\"', '"').replace("\\n", "\n").strip()
