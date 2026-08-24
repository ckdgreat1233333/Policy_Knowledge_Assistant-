"""Policy & Knowledge Management Assistant - Backend API.

Insurance domain. Two deliberately separated interaction flows:

  Business track : underwriters & service teams get clause-level retrieval,
                   a full ReAct reasoning trace, citations, confidence and
                   human-in-the-loop escalation.
  Customer track : customers get simplified, compliance-safe explanations,
                   policy discovery, comparison and renewal guidance.
"""
import hashlib
import logging
import os
import secrets
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import database as db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("policy_assistant")

app = FastAPI(
    title="Policy & Knowledge Management Assistant",
    description="Enterprise policy knowledge system for insurance: RAG + custom "
                "ReAct agent with freshness controls, citations and escalation.",
    version="1.0.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,
                   allow_methods=["*"], allow_headers=["*"])

service = None
try:
    from services.policy_service import get_policy_service
    service = get_policy_service()
except Exception as exc:
    logger.warning("Policy service unavailable: %s", exc)


def _hash_pw(p: str) -> str:
    return hashlib.sha256(p.encode()).hexdigest()


def _verify_pw(p: str, h: str) -> bool:
    return _hash_pw(p) == h


def _token() -> str:
    return f"tok-{secrets.token_hex(16)}"


def _audit(actor, event, details, risk="low"):
    db.create_audit_log(f"LOG-{uuid.uuid4().hex[:8]}", actor, event, details, risk)


# ════════════════ REQUEST MODELS ════════════════

class LoginRequest(BaseModel):
    username: str
    password: str
    portalType: str = "customer"


class RegisterRequest(BaseModel):
    username: str
    fullName: str = ""
    email: str = ""
    password: str
    phone: Optional[str] = ""
    portalType: str = "customer"


class BusinessQueryRequest(BaseModel):
    question: str


class CustomerChatRequest(BaseModel):
    message: str


class CompareRequest(BaseModel):
    policy_a: str
    policy_b: str


# ════════════════ ROOT / HEALTH ════════════════

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    idx = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(idx):
        with open(idx, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read(), headers={
                "Cache-Control": "no-store, no-cache, must-revalidate"})
    return {"service": "Policy & Knowledge Management Assistant", "version": "1.0.0"}


@app.get("/api/health")
async def health():
    llm_ok = False
    try:
        from services.llm_service import LLMService
        llm_ok = LLMService().health_check()
    except Exception:
        pass
    return {"status": "ok", "knowledge_ready": bool(service), "llm_connected": llm_ok}


# ════════════════ AUTH (portal selection separates flows) ════════════════

@app.post("/api/auth/register")
async def register(req: RegisterRequest):
    if db.user_exists(req.username):
        raise HTTPException(400, "Username already taken")
    if len(req.password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")
    db.create_user(req.username, req.email or "", req.fullName or req.username,
                   req.phone or "", _hash_pw(req.password), req.portalType)
    user = db.get_user(req.username)
    _audit(req.username, "User Registration", f"portal={req.portalType}")
    return {"user": {k: v for k, v in user.items() if k != "password"},
            "token": _token()}


@app.post("/api/auth/login")
async def login(req: LoginRequest):
    user = db.get_user(req.username)
    if not user or not _verify_pw(req.password, user.get("password", "")):
        raise HTTPException(401, "Invalid username or password")
    user["role"] = req.portalType
    _audit(req.username, "User Login", f"portal={req.portalType}")
    return {"user": {k: v for k, v in user.items() if k != "password"},
            "token": _token()}


@app.post("/api/auth/logout")
async def logout():
    return {"success": True}


# ════════════════ BUSINESS TRACK ════════════════

@app.post("/api/business/query")
async def business_query(req: BusinessQueryRequest):
    """Underwriter query -> ReAct trace + cited answer + escalation status."""
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(400, "Question is required")
    if service is None:
        raise HTTPException(503, "Policy service unavailable")
    answer = service.business_query(q)
    payload = answer.to_dict()
    _audit("underwriter", "Business Policy Query",
           f"{q[:120]} -> {payload['decision']} conf={payload['confidence']}",
           "high" if payload["needs_human_override"] else "low")
    return payload


# ════════════════ CUSTOMER TRACK ════════════════

@app.post("/api/customer/chat")
async def customer_chat(req: CustomerChatRequest):
    """Customer question -> simplified compliance-safe explanation."""
    msg = (req.message or "").strip()
    if not msg:
        raise HTTPException(400, "Message is required")
    if service is None:
        raise HTTPException(503, "Policy service unavailable")
    answer = service.customer_chat(msg)
    payload = answer if isinstance(answer, dict) else answer.to_dict()
    _audit("customer", "Customer Policy Chat",
           f"{msg[:120]} -> {payload['decision']}",
           "high" if payload["decision"] != "answer" else "low")
    return payload


@app.post("/api/customer/compare")
async def customer_compare(req: CompareRequest):
    """Embedding-aligned comparison of two policies + disclosures."""
    if service is None:
        raise HTTPException(503, "Policy service unavailable")
    try:
        result = service.compare_policies(req.policy_a.strip(), req.policy_b.strip())
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    _audit("customer", "Policy Comparison",
           f"{req.policy_a} vs {req.policy_b}")
    return result


# ════════════════ KNOWLEDGE / FRESHNESS / AUDIT ════════════════

@app.get("/api/policies")
async def list_policies():
    if service is None:
        raise HTTPException(503, "Policy service unavailable")
    docs = [d.to_dict() for d in service.store.documents()]
    return {"policies": [d for d in docs if d["doc_type"] in ("policy", "endorsement")]}


@app.post("/api/knowledge/reingest")
async def knowledge_reingest():
    """Re-scan data/policies/ and rebuild the index for changed/new files
    without restarting the server."""
    if service is None:
        raise HTTPException(503, "Policy service unavailable")
    before = len(service.store.chunks)
    docs_before = {d.document_id for d in service.store.documents()}
    try:
        service.store.ingest()
    except Exception as exc:
        raise HTTPException(500, f"Re-ingestion failed: {exc}")
    docs = [d.to_dict() for d in service.store.documents()]
    added = [d["document_id"] for d in docs if d["document_id"] not in docs_before]
    _audit("underwriter", "Knowledge Re-ingest",
           f"chunks {before}->{len(service.store.chunks)}; added={added or 'none'}")
    return {"chunks": len(service.store.chunks), "documents": len(docs),
            "added": added}


@app.get("/api/knowledge/freshness")
async def knowledge_freshness():
    """Freshness dashboard: version tags, ages, stale warnings, change log."""
    if service is None:
        raise HTTPException(503, "Policy service unavailable")
    dashboard = service.freshness_dashboard()
    dashboard["updates"] = db.list_knowledge_updates()
    return dashboard


@app.get("/api/sessions")
async def list_sessions(limit: int = 50):
    return {"sessions": db.list_policy_sessions(limit=limit)}


@app.get("/api/audit-logs")
async def list_audit_logs(riskLevel: str = ""):
    return {"logs": db.list_audit_logs(risk_level=riskLevel)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
