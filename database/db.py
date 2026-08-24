"""SQLite persistence layer.

Stores portal users, the immutable audit ledger, policy Q&A session history,
and the knowledge update log (mirrored from data/update_log.md at ingest).
"""
import csv
import os
import sqlite3
from datetime import datetime
from typing import Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(os.path.dirname(__file__), "data", "policy_assistant.db")
USERS_CSV = os.path.join(BASE_DIR, "data", "users", "seed_users.csv")


def get_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            email TEXT,
            name TEXT NOT NULL,
            phone TEXT DEFAULT '',
            password TEXT NOT NULL,
            role TEXT DEFAULT 'customer'
        );
        CREATE TABLE IF NOT EXISTS audit_logs (
            id TEXT PRIMARY KEY,
            timestamp TEXT DEFAULT (datetime('now')),
            actor TEXT NOT NULL,
            eventType TEXT NOT NULL,
            riskLevel TEXT DEFAULT 'low',
            details TEXT DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS policy_sessions (
            session_id TEXT PRIMARY KEY,
            timestamp TEXT DEFAULT (datetime('now')),
            track TEXT NOT NULL,
            question TEXT DEFAULT '',
            decision TEXT DEFAULT 'answer',
            intent TEXT DEFAULT '',
            answered INTEGER DEFAULT 0,
            needs_human_override INTEGER DEFAULT 0,
            escalation_reason TEXT DEFAULT '',
            payload TEXT DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS knowledge_updates (
            change_date TEXT,
            document_id TEXT,
            from_version TEXT,
            summary TEXT,
            approved_by TEXT,
            PRIMARY KEY (change_date, document_id)
        );
    """)
    conn.commit()
    conn.close()
    _seed_users()


def _seed_users():
    import hashlib
    conn = get_db()
    count = conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()["c"]
    if count == 0 and os.path.exists(USERS_CSV):
        with open(USERS_CSV, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            r["password"] = hashlib.sha256(
                (r.get("password") or "").encode()).hexdigest()
            conn.execute(
                "INSERT OR REPLACE INTO users VALUES (:username,:email,:name,:phone,:password,:role)",
                r)
        conn.commit()
    conn.close()


# ─── Users ───

def user_exists(username: str) -> bool:
    conn = get_db()
    row = conn.execute("SELECT 1 FROM users WHERE username=?", (username,)).fetchone()
    conn.close()
    return row is not None


def get_user(username_or_email: str) -> Optional[dict]:
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE username=? OR email=?",
                       (username_or_email, username_or_email)).fetchone()
    conn.close()
    return dict(row) if row else None


def create_user(username: str, email: str, name: str, phone: str,
                password: str, role: str = "customer"):
    conn = get_db()
    conn.execute("INSERT OR REPLACE INTO users VALUES (?,?,?,?,?,?)",
                 (username, email, name, phone, password, role))
    conn.commit()
    conn.close()


# ─── Audit ledger ───

def create_audit_log(entry_id: str, actor: str, event_type: str,
                     details: str, risk_level: str = "low"):
    conn = get_db()
    conn.execute(
        "INSERT INTO audit_logs (id, actor, eventType, riskLevel, details) VALUES (?,?,?,?,?)",
        (entry_id, actor, event_type, risk_level, details))
    conn.commit()
    conn.close()


def list_audit_logs(risk_level: str = "") -> list:
    conn = get_db()
    query = "SELECT * FROM audit_logs WHERE 1=1"
    params = []
    if risk_level:
        query += " AND riskLevel=?"
        params.append(risk_level)
    rows = conn.execute(query + " ORDER BY timestamp DESC LIMIT 200",
                        params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ─── Policy sessions ───

def create_policy_session(session_id: str, track: str, question: str,
                          decision: str, intent: str, answered: bool,
                          needs_human_override: bool, escalation_reason: str,
                          payload: str):
    conn = get_db()
    conn.execute(
        "INSERT INTO policy_sessions VALUES (?,?,datetime('now'),?,?,?,?,?,?,?)",
        (session_id, track, question, decision, intent, int(answered),
         int(needs_human_override), escalation_reason or "", payload))
    conn.commit()
    conn.close()


def list_policy_sessions(limit: int = 100) -> list:
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM policy_sessions ORDER BY timestamp DESC LIMIT ?",
        (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def record_knowledge_update(change_date: str, document_id: str,
                            from_version: str, summary: str, approved_by: str):
    conn = get_db()
    conn.execute(
        "INSERT OR REPLACE INTO knowledge_updates VALUES (?,?,?,?,?)",
        (change_date, document_id, from_version, summary, approved_by))
    conn.commit()
    conn.close()


def list_knowledge_updates(limit: int = 50) -> list:
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM knowledge_updates ORDER BY change_date DESC LIMIT ?",
        (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


init_db()
