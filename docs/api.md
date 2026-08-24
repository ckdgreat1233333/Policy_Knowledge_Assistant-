# API Reference

Base URL: `http://localhost:8000`

## Auth

### POST /api/auth/register
```json
{ "username": "newu", "fullName": "New User", "email": "n@x.com",
  "password": "secret1", "portalType": "underwriter" }
```
→ `{ "user": {...no password...}, "token": "tok-..." }`

### POST /api/auth/login
```json
{ "username": "underwriter", "password": "underwriter123", "portalType": "underwriter" }
```
→ `{ "user": { "username", "name", "role", ... }, "token": "tok-..." }`

`portalType` selects the portal: `underwriter` (business track) or `customer`.

## Business track

### POST /api/business/query
Request: `{ "question": "Is maternity covered under SecurePlus and after what waiting period?" }`

Response:
```json
{
  "track": "business",
  "decision": "answer | escalate | refuse",
  "intent": "coverage | exclusion | endorsement | renewal | claim_process | eligibility | general | out_of_scope",
  "answered": true,
  "narrative": "...source-based interpretation with inline [doc §ref] citations...",
  "confidence": 0.94,
  "citations": [
    { "document_id": "pol_health_secureplus", "clause_ref": "3.2",
      "title": "SecurePlus Family Health Policy", "version": "v2.1",
      "effective_date": "2026-05-01", "similarity": 0.71,
      "display": "SecurePlus Family Health Policy, §3.2 (v2.1)" }
  ],
  "reasoning_trace": [
    { "step_no": 1, "thought": "...", "action": "search_clauses",
      "action_input": "...", "observation": "..." }
  ],
  "retrieved_clauses": [ { "clause_ref", "document_id", "title", "doc_type",
      "product_line", "version", "effective_date", "similarity",
      "freshness", "text" } ],
  "disclaimers": ["Internal decision support only. ..."],
  "stale_warnings": ["WARNING/CRITICAL ..."],
  "compliance_notes": ["Grounded on N validated clause citation(s); ..."],
  "needs_human_override": false,
  "escalation_reason": "",
  "suggested_followups": ["...", "..."]
}
```

## Customer track

### POST /api/customer/chat
Request: `{ "message": "What happens if I miss my renewal date?" }`

Response: same shape as business but with
- `reasoning_trace: []` and `retrieved_clauses: []` (internal detail never exposed),
- customer disclaimers (not legal advice; policy document prevails; confirm with team),
- `citations` reduced to source chips,
- `suggested_followups` next questions.

### POST /api/customer/compare
```json
{ "policy_a": "SecurePlus Health", "policy_b": "DriveCare Motor" }
```
```json
{
  "policy_a": { "id": "pol_health_secureplus", "title": "..." },
  "policy_b": { "id": "pol_motor_drivecare", "title": "..." },
  "rows": [ { "topic": "Scope Of Cover", "text_a": "...", "text_b": "...",
              "similarity": 0.62, "relation": "same|different|only_a|only_b" } ],
  "summary": "factual comparison narrative",
  "disclosures": ["similarity ≠ legal equivalence", "pricing excluded", "..."]
}
```

## Knowledge

### GET /api/knowledge/freshness
```json
{ "documents": [ { "document_id", "title", "doc_type", "product_line",
    "version", "effective_date", "supersedes", "age_days",
    "freshness_status": "fresh|stale|critical", "clause_count", "last_change" } ],
  "counts": { "fresh": 6, "stale": 0, "critical": 1 },
  "updates": [ { "change_date", "document_id", "from_version",
                 "summary", "approved_by" } ] }
```

### GET /api/policies
Ingested documents of type `policy|endorsement` with metadata.

## Audit

### GET /api/sessions?limit=50
Session history: timestamp, track, question, intent, decision, override flag, escalation reason.

### GET /api/audit-logs?riskLevel=high
Immutable audit ledger entries.

### GET /api/health
`{ "status", "knowledge_ready", "llm_connected" }`
