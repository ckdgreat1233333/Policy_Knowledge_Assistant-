"""Data models for the Policy & Knowledge Management Assistant."""
from dataclasses import dataclass, field


@dataclass
class PolicyChunk:
    """A clause-level chunk of a policy corpus document with full provenance."""
    chunk_id: str
    text: str
    clause_ref: str
    document_id: str
    title: str
    doc_type: str = "policy"          # policy | endorsement | sop | faq
    product_line: str = ""            # health | motor | life | all | internal
    policy_id: str = ""
    version: str = "v1.0"
    effective_date: str = ""          # ISO date string
    supersedes: str = ""

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id, "text": self.text,
            "clause_ref": self.clause_ref, "document_id": self.document_id,
            "title": self.title, "doc_type": self.doc_type,
            "product_line": self.product_line, "policy_id": self.policy_id,
            "version": self.version, "effective_date": self.effective_date,
            "supersedes": self.supersedes,
        }


@dataclass
class RetrievedClause:
    """A retrieved chunk together with its similarity score and freshness status."""
    chunk: PolicyChunk
    similarity: float
    freshness_status: str = "fresh"   # fresh | stale | critical

    def citation_text(self, max_len: int = 400) -> str:
        text = self.chunk.text
        return text if len(text) <= max_len else text[:max_len].rsplit(" ", 1)[0] + "..."

    def to_dict(self) -> dict:
        return {
            "clause_ref": self.chunk.clause_ref,
            "document_id": self.chunk.document_id,
            "title": self.chunk.title,
            "doc_type": self.chunk.doc_type,
            "product_line": self.chunk.product_line,
            "version": self.chunk.version,
            "effective_date": self.chunk.effective_date,
            "similarity": round(self.similarity, 4),
            "freshness": self.freshness_status,
            "text": self.chunk.text,
        }

    @property
    def ref(self) -> str:
        return f"{self.chunk.document_id} §{self.chunk.clause_ref} ({self.chunk.version})"


@dataclass
class AgentStep:
    """One Thought / Action / Observation cycle of the ReAct trace."""
    step_no: int
    thought: str
    action: str
    action_input: str = ""
    observation: str = ""

    def to_dict(self) -> dict:
        return {
            "step_no": self.step_no,
            "thought": self.thought,
            "action": self.action,
            "action_input": self.action_input,
            "observation": self.observation,
        }


@dataclass
class Citation:
    """A validated source reference attached to a generated answer."""
    document_id: str
    title: str
    clause_ref: str
    version: str
    effective_date: str
    similarity: float

    @property
    def display(self) -> str:
        return f"{self.title}, §{self.clause_ref} ({self.version})"

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id, "title": self.title,
            "clause_ref": self.clause_ref, "display": self.display,
            "version": self.version, "effective_date": self.effective_date,
            "similarity": round(self.similarity, 4),
        }


@dataclass
class PolicyAnswer:
    """Structured, fully-traceable answer for either interaction track.
    track is 'business' (underwriter/service teams) or 'customer'."""
    track: str
    question: str
    answered: bool
    decision: str                     # answer | escalate | refuse
    narrative: str
    intent: str = ""
    confidence: float = 0.0
    citations: list[Citation] = field(default_factory=list)
    reasoning_trace: list[AgentStep] = field(default_factory=list)
    retrieved_clauses: list[RetrievedClause] = field(default_factory=list)
    disclaimers: list[str] = field(default_factory=list)
    stale_warnings: list[str] = field(default_factory=list)
    compliance_notes: list[str] = field(default_factory=list)
    needs_human_override: bool = False
    escalation_reason: str = ""
    suggested_followups: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "track": self.track, "question": self.question,
            "answered": self.answered, "decision": self.decision,
            "intent": self.intent, "narrative": self.narrative,
            "confidence": round(self.confidence, 3),
            "citations": [c.to_dict() for c in self.citations],
            "reasoning_trace": [s.to_dict() for s in self.reasoning_trace],
            "retrieved_clauses": [r.to_dict() for r in self.retrieved_clauses],
            "disclaimers": self.disclaimers,
            "stale_warnings": self.stale_warnings,
            "compliance_notes": self.compliance_notes,
            "needs_human_override": self.needs_human_override,
            "escalation_reason": self.escalation_reason,
            "suggested_followups": self.suggested_followups,
        }


@dataclass
class ComparisonRow:
    """One aligned topic across two policies from the embedding comparison."""
    topic: str
    text_a: str
    text_b: str
    similarity: float                 # 1.0 = identical meaning, lower = divergent
    relation: str                     # same | different | only_a | only_b

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "text_a": self.text_a,
            "text_b": self.text_b,
            "similarity": round(self.similarity, 3),
            "relation": self.relation,
        }


@dataclass
class ComparisonResult:
    """Embedding-aligned comparison of two policies plus LLM summary."""
    policy_a_id: str
    policy_b_id: str
    title_a: str
    title_b: str
    rows: list[ComparisonRow] = field(default_factory=list)
    summary: str = ""
    disclosures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "policy_a": {"id": self.policy_a_id, "title": self.title_a},
            "policy_b": {"id": self.policy_b_id, "title": self.title_b},
            "rows": [r.to_dict() for r in self.rows],
            "summary": self.summary,
            "disclosures": self.disclosures,
        }


@dataclass
class DocumentInfo:
    """Freshness metadata for one ingested corpus document."""
    document_id: str
    title: str
    doc_type: str
    product_line: str
    version: str
    effective_date: str
    supersedes: str
    age_days: int
    freshness_status: str             # fresh | stale | critical
    clause_count: int
    last_change: str = ""             # most recent update-log entry for this doc

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id, "title": self.title,
            "doc_type": self.doc_type, "product_line": self.product_line,
            "version": self.version, "effective_date": self.effective_date,
            "supersedes": self.supersedes, "age_days": self.age_days,
            "freshness_status": self.freshness_status,
            "clause_count": self.clause_count, "last_change": self.last_change,
        }
