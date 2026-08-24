# Ethics, Risk & Governance Controls

## 1. Hallucination prevention (layered)

| Layer | Control | Where |
|-------|---------|-------|
| Retrieval gate | Hits below similarity 0.30 discarded; empty evidence never reaches the LLM — it escalates instead | `knowledge/policy_store.retrieve` |
| Grounded prompting | "Use only facts present in the clauses. No outside insurance knowledge." in every generation template | `prompts/*_answer_prompt.txt` |
| Citation validation | Every `[doc §ref]` emitted by the LLM must resolve to an actually-retrieved chunk; unresolvable refs are dropped and lower confidence | `agents/policy_agent._validate_citations` |
| Output contract | Strict JSON answers; truncated/invalid output is salvaged or replaced by a source-listing fallback marked `_degraded` → forced escalation | `agents/policy_agent._generate_answer` |
| Deterministic decision | answer / escalate / refuse decided by code thresholds, never by the model | `agents/policy_agent._decide` |

## 2. Over-simplification risks (customer track)

Simplified language can drop conditions that matter. Mitigations:
- Every customer reply carries the disclaimer: *"The policy document, its wordings and endorsements prevail over this summary."*
- Source chips show exactly which clause versions informed the summary.
- Numbers are preserved verbatim from clauses (limits, waiting periods); the prompt forbids inventing figures.
- The safe-language scrub rewrites any absolute promise ("all expenses covered", "no waiting period") into conditional wording.

## 3. Policy misinterpretation impact

- **Version correctness**: every retrieved clause displays its version + effective date; answers name which document version governs.
- **Precedence**: SOP-UW-011 order (regulatory > endorsement > base wording) is embedded into business-track prompts.
- **Ambiguity surfacing**: the generator must list `ambiguities` instead of guessing; these surface as warnings on the underwriter console.
- **Conflict escalation**: materially conflicting or incomplete evidence forces escalation with a documented reason.

## 4. Knowledge freshness

- `fresh ≤365d` · `stale >365d` (warning attached) · `critical >730d` (auto-escalation). Undated documents are treated as critical.
- Confidence scores are penalized for stale/critical sources so downstream consumers can trust the number.
- Append-only update log provides audit-grade change history (date, doc, change, approver).

## 5. Scope discipline

Refused outright (routed to humans):
- Legal opinions, litigation strategy, tribunal outcome predictions.
- Claim approval/rejection pre-judgement.
- Pricing/quotation decisions.

The refusal message explains *why* and redirects to the correct channel — the assistant stays useful without overstepping.

## 6. Human-in-the-loop

- Escalations are first-class outcomes with explicit reasons shown in the UI and stored in `policy_sessions`.
- Business track marks `needs_human_override=true`; the console banner instructs senior-underwriter sign-off per SOP timelines (ack 4h, resolve 24h).
- Customer track never exposes internal reasoning; unresolved cases hand off to the service desk with a clear message.

## 7. Accountability & auditability

- Immutable `audit_logs` ledger (logins, queries, comparisons, risk levels).
- `policy_sessions` stores question, intent, decision, confidence, citations count and escalation reason for every interaction.
- Compliance notes attached to each response record post-generation scrubs that were applied.

## 8. Known limitations & improvements

1. **Heuristic intent classifier** — could be replaced by a fine-tuned classifier trained on labelled underwriter queries.
2. **Single embedding space for all doc types** — doc-type-aware retrieval (e.g., separate namespaces per product line) would sharpen recall.
3. **No reranker** — a cross-encoder rerank stage would improve precision on multi-topic questions.
4. **English-only corpus** — multilingual policies need language detection + per-language indexes.
5. **Escalation workflow is advisory** — integrating a ticketing system with SLA tracking would close the loop operationally.
6. **Evaluation harness** — golden Q/A set per policy line with automated grounding/recall scoring should run in CI.
