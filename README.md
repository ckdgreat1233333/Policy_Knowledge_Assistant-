# Policy & Knowledge Management Assistant (Insurance Domain)

An enterprise-grade **Policy & Knowledge Management Assistant** that retrieves,
reasons over and explains insurance policies safely — for **underwriters and
service teams (business track)** and for **customers (customer track)**.
Built on GenAI + RAG + a custom ReAct agent, with knowledge freshness controls,
clause-level citations, confidence scoring and human-in-the-loop escalation.

> In insurance, accuracy beats creativity — every time.

---

## What it does

| Track | Audience | What you get |
|-------|----------|--------------|
| **Business track** | Underwriters, service teams | Clause-level semantic retrieval, full ReAct reasoning trace (Thought/Action/Observation), exact citations with version tags, confidence score, ambiguity detection and escalation to senior underwriters |
| **Customer track** | Policyholders | Simplified explanations of coverage/exclusions/renewals/endorsements, embedding-based policy comparison, mandatory compliance disclaimers, zero internal reasoning exposure |

Out of scope by design: legal advice, policy issuance/pricing decisions,
claims adjudication.

### Core principle: retrieval decides, code validates, the LLM only explains

```
PolicyKnowledgeStore (RAG)     → clause-level evidence with version tags
        │  similarity-threshold gated; empty context ⇒ escalate, never guess
        ▼
PolicyReActAgent               → Thought → Action → Observation loop
        │  completeness + freshness validated each turn
        ▼
Grounded generation            → strict JSON, citations validated against evidence
        ▼
Deterministic decision         → answer | escalate | refuse  (never "creative")
        ▼
Compliance scrubber            → non-guarantee language enforced AFTER the LLM
```

If the LLM fails or drifts, the system degrades to deterministic template
answers or escalation — it never invents policy content.

---

## Architecture (RAG + Agent)

```
┌────────────────────────────────────────────────────────────────────────────┐
│                     CLIENT LAYER (Vanilla JS SPA, static/)                 │
│   ┌───────────────────────────────┐   ┌────────────────────────────────┐   │
│   │ UNDERWRITER CONSOLE           │   │ CUSTOMER PORTAL                │   │
│   │ · query console               │   │ · simplified Q&A chat          │   │
│   │ · ReAct trace timeline        │   │ · embedding policy comparison  │   │
│   │ · citations + confidence      │   │ · disclaimers on every answer  │   │
│   │ · freshness dashboard         │   │                                │   │
│   └──────────────┬────────────────┘   └──────────────┬─────────────────┘   │
└──────────────────┼────────────────────────────────────┼────────────────────┘
                   ▼                                    ▼
        ┌──────────────── FastAPI (app.py) — separated flows ─────────────┐
        │ /api/business/query        /api/customer/chat                  │
        │ /api/knowledge/freshness   /api/customer/compare               │
        │ auth (portal roles) · audit ledger · session history            │
        └──────────────────────────────┬──────────────────────────────────┘
                                       ▼
                    ┌─────────────────────────────────────┐
                    │  PolicyService (facade)             │
                    │  scope guardrails → agent → scrub   │
                    └───────┬───────────────┬─────────────┘
                            ▼               ▼
      ┌──────────────────────────┐   ┌───────────────────────────────┐
      │ KNOWLEDGE LAYER          │   │ AGENT LAYER                   │
      │ data/policies/*.txt      │   │ agents/policy_agent.py        │
      │  ↓ sha256 hash manifest  │   │ custom ReAct:                 │
      │ chunker.py (clauses)     │   │  Thought: intent + gaps       │
      │ embedder.py (MiniLM)     │   │  Action: search_clauses /     │
      │ FAISS IndexFlatL2        │   │          inspect_document /   │
      │ freshness.py             │   │          finish               │
      │  fresh/stale/critical    │   │  Observation: validate        │
      │ comparator.py (compare)  │   │  Decision: answer/escalate    │
      └──────────────────────────┘   └───────────────────────────────┘
                            ▼                       ▼
      ┌──────────────────────────────────────────────────────────────┐
      │ GUARDRAILS: scope refusal (legal/claim adjudication)         │
      │ post-generation safe-language scrub, citation validation,    │
      │ confidence scoring, stale-document warnings                  │
      └──────────────────────────────────────────────────────────────┘
                            ▼
      SQLite: users · audit_logs · policy_sessions · knowledge_updates
```

### Data flow (one business-track query)

1. **Scope check** – legal-advice / claim-adjudication requests are refused outright.
2. **Intent** – heuristic classifier (coverage / exclusion / endorsement / renewal / claim_process / eligibility).
3. **ReAct step** – LLM emits `{"thought", "action", "action_input"}`; tools run against the vector store.
4. **Observation validation** – similarity threshold, clause coverage count, per-document version & age.
5. **Loop or stop** – refine query when evidence weak; early-finish when evidence is strong; max 4 steps.
6. **Grounded generation** – strict JSON over retrieved clauses only.
7. **Citation validation** – every `[doc §ref]` is resolved against actual retrieved chunks; unresolvable citations are dropped.
8. **Confidence scoring** – f(top similarities, evidence count, freshness penalties, degradation flags); < 0.45 ⇒ escalate.
9. **Decision** – answer / escalate (human-in-the-loop) / refuse; everything audited.

---

## Knowledge ingestion & freshness controls

- **Corpus**: `data/policies/*.txt` — health, motor and life policy wordings, endorsements, an internal SOP and customer FAQs. Every file carries a header (`Doc Type`, `Product Line`, `Version`, `Effective Date`, `Supersedes`).
- **Ingestion**: sha256 content hash manifest → clause chunking → MiniLM embeddings → FAISS index + JSON metadata sidecar. The index rebuilds only when files change (fully reproducible).
- **Chunking strategy**: clause-boundary chunking. Insurance clauses are self-contained citable units — this keeps chunks coherent, gives exact citation refs (§4, §3.2), avoids mid-clause truncation, and long clauses are sentence-split under a size cap so embeddings stay topical.
- **Embedding model**: `all-MiniLM-L6-v2` (384-dim, normalized) — fast CPU inference, strong on short technical text; L2 distance maps directly to cosine similarity.
- **Similarity handling**: hits below 0.30 are discarded *before* generation; an empty context escalates rather than inviting hallucination.
- **Freshness rules**: ≤365 days = `fresh`; >365 = `stale` (warning attached to answers); >730 = `critical` (answers relying on it are escalated to a senior underwriter). Undated documents are treated as critical.
- **Update log**: append-only `data/update_log.md` feeds the freshness dashboard ("what changed, when, approved by whom").

---

## Ethics, risk & governance controls

| Risk | Control |
|------|---------|
| Hallucination | Retrieval-threshold gating, grounded-only prompts, citation validation against evidence, template fallback, no-evidence ⇒ escalate |
| Misinterpretation | SOP precedence encoded (regulatory > endorsement > base wording), version tags surfaced in every UI panel |
| Stale knowledge | Version + effective-date tracking, stale warnings, automatic escalation past review window |
| Over-simplification (customers) | "Policy document prevails" disclaimer on every reply, source chips shown, no guarantees language |
| Legal exposure | Scope refusals for legal advice and claim adjudication, routed to humans |
| Non-guarantee claims | Banned-pattern scrub applied AFTER generation; violations logged in response |
| Accountability | Immutable audit ledger, session history incl. decision/confidence/escalation reason |

Full narrative: `docs/governance.md`.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | Python 3.12, FastAPI, Uvicorn |
| Frontend | Vanilla JS SPA (`static/index.html`) |
| LLM | Groq via OpenAI SDK (`openai/gpt-oss-120b`) |
| Embeddings | Sentence Transformers `all-MiniLM-L6-v2` |
| Vector store | FAISS `IndexFlatL2` (cosine via normalized vectors) |
| Database | SQLite (`database/data/policy_assistant.db`) |
| Auth/Audit | Role portals (underwriter/customer), immutable ledger |

---

## Project Structure

```
├── app.py                          # FastAPI entry point, both tracks' endpoints
├── config.py                       # LLM provider / model / key
├── data/
│   ├── policies/                   # versioned corpus (policies, endorsements, SOP, FAQ)
│   ├── update_log.md               # append-only knowledge change log
│   ├── users/seed_users.csv        # seeded portal accounts
│   └── indexes/                    # FAISS index + chunk metadata sidecar
├── knowledge/
│   ├── chunker.py                  # clause chunking + header parsing
│   ├── embedder.py                 # shared MiniLM wrapper
│   ├── policy_store.py             # ingestion, versioning, threshold-gated retrieval
│   ├── freshness.py                # age/status rules + update-log parsing
│   └── comparator.py              # embedding-aligned policy comparison
├── agents/
│   └── policy_agent.py             # custom ReAct loop, citation validation, decisions
├── guardrails/
│   └── compliance.py               # scope refusals + safe-language scrub + disclaimers
├── services/
│   ├── policy_service.py           # facade wiring both tracks + comparison + logging
│   ├── llm_service.py              # Groq/OpenAI client wrapper
│   └── prompt_service.py           # prompt template loader
├── models/policy.py                # chunks, clauses, trace steps, answers, comparisons
├── database/                       # db.py (SQLite) + faiss_db.py
├── prompts/                        # react_step, business_answer, customer_answer, comparison
├── static/index.html               # underwriter console + customer portal SPA
├── tests/test_policy.py            # 19 tests (chunking, RAG, agent, guardrails, service)
└── docs/                           # architecture.md, governance.md, api.md
```

---

## Quick Start

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# .env
# GROQ_API_KEY=your-groq-api-key

uvicorn app:app --reload --port 8000
```

Open `http://localhost:8000`. Seeded logins:

| Portal | Username | Password |
|--------|----------|----------|
| Underwriter (business track) | `underwriter` | `underwriter123` |
| Customer (customer track) | `customer` | `customer123` |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/business/query` | **Business track**: `{question}` → decision, ReAct trace, validated citations, confidence, escalation reason, stale warnings |
| POST | `/api/customer/chat` | **Customer track**: `{message}` → simplified explanation, source chips, disclaimers, follow-ups |
| POST | `/api/customer/compare` | `{policy_a, policy_b}` → embedding-aligned clause rows + factual summary + disclosures |
| GET | `/api/knowledge/freshness` | Freshness dashboard: versions, ages, statuses, update log |
| GET | `/api/policies` | Ingested policy/endorsement documents |
| GET | `/api/sessions` | Session history (decision, confidence, escalations) |
| GET | `/api/audit-logs` | Immutable audit ledger |
| POST | `/api/auth/login·register` | Portal auth (`portalType`: underwriter/customer) |

Details: `docs/api.md`.

---

## Demo walkthroughs (evaluation-ready)

1. **Underwriter coverage query** — "Is maternity covered under SecurePlus and after what waiting period?" → intent=coverage, retrieves §3.2 + maternity endorsement, shows precedence-aware answer with citations, confidence ~0.97.
2. **Stale-knowledge escalation** — any LifeShield question → document flagged CRITICAL (>730 days) → auto-escalation to senior underwriter with reason.
3. **Refusal** — "Should I go to court over this claim dispute?" → refused with routing guidance (out-of-scope guardrail).
4. **Customer explanation** — renewal grace period question → simple language, source chips, three compliance disclaimers.
5. **Comparison** — SecurePlus vs DriveCare → embedding-aligned clause rows (similar/different/only-in-one) + factual summary + limitation disclosures.
6. **Freshness dashboard** — corpus table with version tags, ages, statuses and the update log.

---

## Tests

```bash
pytest tests/ -q
```

19 tests covering clause chunking, header parsing, freshness thresholds, update-log parsing, corpus grounding, similarity-threshold noise rejection, intent classification, ReAct trace + citation validation, stale-doc escalation penalty, safe-language scrubbing, scope refusals, both tracks' response contracts and the comparison engine (LLM stubbed — no network needed).

## Presentation map

| Capstone expectation | Where |
|----------------------|-------|
| Problem framing | README intro + docs/architecture.md |
| Ingestion & RAG flow | knowledge/ + diagram above |
| ReAct reasoning walkthrough | Underwriter console trace panel + agents/policy_agent.py |
| Internal vs customer responses | Two portals, deliberately different payloads |
| Risk, ethics & compliance | guardrails/, docs/governance.md, compliance notes in responses |
| Challenges & improvements | docs/governance.md §limitations |
