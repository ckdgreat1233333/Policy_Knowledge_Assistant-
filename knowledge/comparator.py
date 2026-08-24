"""Semantic policy comparison using embeddings.

Method:
  1. every clause of both documents is embedded (same normalized MiniLM space),
  2. for each clause of A the best-matching clause of B is found by cosine
     similarity,
  3. pairs are labelled: same (>=0.75) / different (0.45-0.75) / only_a (<0.45),
  4. unmatched B-clauses surface as only_b,
  5. an LLM writes a factual narrative strictly over these pairs.

The similarity labels are deterministic - the LLM never decides what is
"different", it only explains the differences already computed.
"""
from __future__ import annotations

import logging

import numpy as np

from models.policy import ComparisonResult, ComparisonRow

logger = logging.getLogger("policy_comparator")

SAME_THRESHOLD = 0.75
DIFFERENT_THRESHOLD = 0.45


def _topic_label(text: str, fallback: str) -> str:
    words = " ".join(text.split()[1:7]).strip(" .,")
    words = words.replace("The ", "").strip()
    return (words or fallback)[:60]


class PolicyComparator:

    def __init__(self, store, llm, prompts):
        self.store = store
        self.llm = llm
        self.prompts = prompts

    def compare(self, id_a: str, id_b: str) -> ComparisonResult:
        doc_a = self.store.find_document(id_a)
        doc_b = self.store.find_document(id_b)
        if not doc_a or not doc_b or doc_a.document_id == doc_b.document_id:
            raise ValueError("Two distinct known policy documents are required.")

        clauses_a = [c for c in self.store.get_document_chunks(doc_a.document_id)]
        clauses_b = [c for c in self.store.get_document_chunks(doc_b.document_id)]
        if not clauses_a or not clauses_b:
            raise ValueError("One of the documents has no ingested clauses.")

        texts_a = [c.chunk.text for c in clauses_a]
        texts_b = [c.chunk.text for c in clauses_b]
        emb_a = self._embed(texts_a)
        emb_b = self._embed(texts_b)

        used_b: set[int] = set()
        rows: list[ComparisonRow] = []
        for i, rc_a in enumerate(clauses_a):
            sims = emb_a[i] @ emb_b.T
            j = int(np.argmax(sims))
            best = float(sims[j])
            if best >= DIFFERENT_THRESHOLD:
                used_b.add(j)
                relation = "same" if best >= SAME_THRESHOLD else "different"
                rows.append(ComparisonRow(
                    topic=_topic_label(rc_a.chunk.text, f"§{rc_a.chunk.clause_ref}"),
                    text_a=self._short(rc_a.chunk.text),
                    text_b=self._short(clauses_b[j].chunk.text),
                    similarity=best, relation=relation))
            else:
                rows.append(ComparisonRow(
                    topic=_topic_label(rc_a.chunk.text, f"§{rc_a.chunk.clause_ref}"),
                    text_a=self._short(rc_a.chunk.text),
                    text_b="", similarity=best, relation="only_a"))

        for j, rc_b in enumerate(clauses_b):
            if j not in used_b and len(rows) < 24:
                rows.append(ComparisonRow(
                    topic=_topic_label(rc_b.chunk.text, f"§{rc_b.chunk.clause_ref}"),
                    text_a="", text_b=self._short(rc_b.chunk.text),
                    similarity=0.0, relation="only_b"))

        result = ComparisonResult(
            policy_a_id=doc_a.document_id, policy_b_id=doc_b.document_id,
            title_a=doc_a.title, title_b=doc_b.title, rows=rows)
        result.summary = self._summarize(rows)
        return result

    # ── Helpers ────────────────────────────────────────────────

    def _embed(self, texts: list[str]) -> np.ndarray:
        return self.store.embedder.embed(texts)

    @staticmethod
    def _short(text: str, limit: int = 260) -> str:
        return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "..."

    def _summarize(self, rows: list[ComparisonRow]) -> str:
        pairs = "\n".join(
            f"TOPIC: {r.topic}\n  A: {r.text_a[:200] or '(not present in A)'}\n"
            f"  B: {r.text_b[:200] or '(not present in B)'}\n"
            f"  ALIGNMENT: {r.relation} ({r.similarity:.2f})"
            for r in rows[:14])
        try:
            prompt = self.prompts.load("comparison_prompt.txt", pairs=pairs)
            raw = self.llm.generate(prompt, temperature=0.2, max_tokens=500)
            from agents.policy_agent import _extract_json
            parsed = _extract_json(raw) or {}
            summary = str(parsed.get("summary") or "").strip()
            if summary:
                return summary
        except Exception as exc:
            logger.warning("Comparison LLM failed: %s", exc)
        same = sum(1 for r in rows if r.relation == "same")
        diff = sum(1 for r in rows if r.relation == "different")
        uniq = sum(1 for r in rows if r.relation.startswith("only"))
        return (f"Clause alignment: {same} closely matched, {diff} materially "
                f"different, {uniq} present in only one document.")
