"""Policy knowledge store: ingestion, versioning, semantic retrieval.

Ingestion & versioning pipeline:
    corpus file -> sha256 hash -> parse header + clause chunks
        -> embed (MiniLM, normalized) -> FAISS IndexFlatL2
        + chunk-metadata JSON sidecar

The index is rebuilt only when a file's content hash changes or files are
added/removed, so the vector store is always reproducible from the corpus.
Every chunk carries document version + effective date, which powers the
freshness controls and clause-level citations in every answer.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path

from database.faiss_db import FAISSDatabase
from knowledge.chunker import CLAUSE_REF_RE, parse_metadata, split_clauses
from knowledge.embedder import Embedder
from knowledge.freshness import (
    age_in_days,
    evaluate_freshness,
    freshness_status,
    latest_change_for,
    parse_update_log,
)
from models.policy import DocumentInfo, PolicyChunk, RetrievedClause

logger = logging.getLogger("policy_kb")

BASE = Path(__file__).resolve().parent.parent
CORPUS_DIR = BASE / "data" / "policies"
INDEX_PATH = BASE / "data" / "indexes" / "policy_index.bin"
META_PATH = BASE / "data" / "indexes" / "policy_chunks.json"

SUPPORTED_EXTENSIONS = {".txt", ".md"}
SIM_THRESHOLD = 0.30          # below this a hit is considered noise


def file_hash(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _read_pdf(path: Path) -> str:
    try:
        import fitz
    except ImportError:
        logger.warning("PyMuPDF not installed; skipping %s", path.name)
        return ""
    with fitz.open(str(path)) as doc:
        return "\n".join(page.get_text() for page in doc)


class PolicyKnowledgeStore:

    def __init__(self, corpus_dir=CORPUS_DIR, index_path=INDEX_PATH,
                 meta_path=META_PATH, embedder=None):
        self.corpus_dir = Path(corpus_dir)
        self.index_path = Path(index_path)
        self.meta_path = Path(meta_path)
        self.embedder = embedder or Embedder()
        self.database = FAISSDatabase()
        self.chunks: list[PolicyChunk] = []
        self._ready = False
        self.ingest()

    # ── Ingestion ──────────────────────────────────────────────

    def ingest(self) -> None:
        docs = self._scan_corpus()
        manifest = self._load_manifest()
        if manifest and self._manifest_matches(manifest, docs):
            try:
                self.chunks = [self._chunk_from_dict(d)
                               for d in manifest.get("chunks", [])]
                self.database.load(str(self.index_path))
                self._ready = self.database.index is not None
                logger.info("Policy store loaded cached index (%d chunks)",
                            len(self.chunks))
                return
            except Exception as exc:
                logger.warning("Cached index load failed: %s", exc)

        self.chunks = []
        for doc_id, path, _doc_hash in docs:
            text = self._read_document(path)
            if not text.strip():
                continue
            meta = parse_metadata(text)
            title = meta.get("title") or doc_id.replace("_", " ").title()
            for clause_text in split_clauses(text):
                m = CLAUSE_REF_RE.match(clause_text)
                if not m:
                    continue  # only numbered clauses become citable chunks
                ref = m.group(1)
                clean = clause_text.strip()
                self.chunks.append(PolicyChunk(
                    chunk_id=f"{doc_id}-{ref.replace('.', '_')}",
                    text=clean,
                    clause_ref=ref,
                    document_id=doc_id,
                    title=title,
                    doc_type=meta.get("doc_type", "policy"),
                    product_line=meta.get("product_line", ""),
                    policy_id=meta.get("policy_id", ""),
                    version=meta.get("version", "v1.0"),
                    effective_date=meta.get("effective_date", ""),
                    supersedes=meta.get("supersedes", ""),
                ))
        if not self.chunks:
            logger.warning("No clauses ingested from %s", self.corpus_dir)
            return
        embeddings = self.embedder.embed([c.text for c in self.chunks])
        self.database.build(embeddings)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.database.save(str(self.index_path))
        self._write_manifest({d: h for d, _, h in docs})
        self._ready = True
        logger.info("Policy store rebuilt: %d clauses from %d documents",
                    len(self.chunks), len(docs))

    def _scan_corpus(self) -> list[tuple[str, Path, str]]:
        if not self.corpus_dir.exists():
            self.corpus_dir.mkdir(parents=True, exist_ok=True)
            return []
        out = []
        for path in sorted(self.corpus_dir.iterdir()):
            if path.is_dir() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            out.append((path.stem, path, file_hash(str(path))))
        return out

    def _read_document(self, path: Path) -> str:
        if path.suffix.lower() == ".pdf":
            return _read_pdf(path)
        for enc in ("utf-8-sig", "utf-8", "utf-16", "latin-1"):
            try:
                return path.read_text(encoding=enc)
            except UnicodeDecodeError:
                continue
            except OSError:
                return ""
        return ""

    # ── Manifest (reproducibility sidecar) ─────────────────────

    def _load_manifest(self) -> dict:
        if not self.meta_path.exists():
            return {}
        try:
            return json.loads(self.meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    @staticmethod
    def _manifest_matches(manifest: dict,
                          docs: list[tuple[str, Path, str]]) -> bool:
        registered = manifest.get("documents", {})
        if not registered or len(registered) != len(docs):
            return False
        return all(registered.get(d) == h for d, _, h in docs)

    def _write_manifest(self, documents: dict) -> None:
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "documents": documents,
            "chunks": [c.to_dict() for c in self.chunks],
        }
        self.meta_path.write_text(json.dumps(payload, ensure_ascii=False),
                                  encoding="utf-8")

    # ── Retrieval ──────────────────────────────────────────────

    def retrieve(self, query: str, k: int = 5,
                 min_similarity: float = SIM_THRESHOLD,
                 doc_type: str = "", product_line: str = "",
                 ) -> list[RetrievedClause]:
        """Semantic retrieval with similarity-threshold filtering.

        Similarity handling: hits below min_similarity are discarded rather
        than passed to the LLM - an empty context escalates instead of
        inviting hallucination.
        """
        if not self._ready or not self.chunks:
            return []
        qvec = self.embedder.embed_query(query)
        fetch_k = max(k * 6, 24)
        distances, indices = self.database.search(qvec, fetch_k)
        results: list[RetrievedClause] = []
        seen: set[str] = set()
        for dist, idx in zip(distances[0], indices[0]):
            idx = int(idx)
            if idx < 0 or idx >= len(self.chunks):
                continue
            sim = max(0.0, min(1.0, 1.0 - (float(dist) * float(dist)) / 2.0))
            if sim < min_similarity:
                continue
            chunk = self.chunks[idx]
            if chunk.chunk_id in seen:
                continue
            if doc_type and chunk.doc_type != doc_type:
                continue
            if product_line and chunk.product_line not in (product_line, "all"):
                continue
            seen.add(chunk.chunk_id)
            results.append(RetrievedClause(
                chunk=chunk, similarity=sim,
                freshness_status=freshness_status(age_in_days(chunk.effective_date)),
            ))
            if len(results) >= k:
                break
        results.sort(key=lambda r: r.similarity, reverse=True)
        return results

    def get_document_chunks(self, document_id: str) -> list[RetrievedClause]:
        """All clauses of one document ordered by clause reference."""
        rows = [c for c in self.chunks if c.document_id == document_id]
        rows.sort(key=lambda c: [int(p) for p in c.clause_ref.split(".")])
        status = freshness_status(age_in_days(rows[0].effective_date)) if rows else "fresh"
        return [RetrievedClause(chunk=c, similarity=1.0, freshness_status=status)
                for c in rows]

    def documents(self) -> list[DocumentInfo]:
        """Freshness dashboard data for every ingested document."""
        from collections import OrderedDict
        changes = parse_update_log(BASE)
        grouped: "OrderedDict[str, list[PolicyChunk]]" = OrderedDict()
        for c in self.chunks:
            grouped.setdefault(c.document_id, []).append(c)
        infos = []
        for doc_id, cl in grouped.items():
            first = cl[0]
            age, status = evaluate_freshness(first.effective_date)
            infos.append(DocumentInfo(
                document_id=doc_id, title=first.title, doc_type=first.doc_type,
                product_line=first.product_line, version=first.version,
                effective_date=first.effective_date, supersedes=first.supersedes,
                age_days=age, freshness_status=status, clause_count=len(cl),
                last_change=latest_change_for(changes, doc_id),
            ))
        return infos

    def find_document(self, name_or_id: str) -> PolicyChunk | None:
        """Resolve a document by id, policy id, title or fuzzy word overlap."""
        import re as _re
        q = (name_or_id or "").strip().lower()
        if not q:
            return None
        for c in self.chunks:
            if q in (c.document_id.lower(), (c.policy_id or "").lower(),
                     c.title.lower()):
                return c
        tokens = [t for t in _re.split(r"\W+", q) if len(t) > 2]
        if not tokens:
            return None
        best, best_score = None, 0.0
        seen_docs = set()
        for c in self.chunks:
            if c.document_id in seen_docs:
                continue
            seen_docs.add(c.document_id)
            hay = f"{c.document_id} {c.title} {c.policy_id}".lower()
            score = sum(1 for t in tokens if t in hay) / len(tokens)
            if score > best_score:
                best, best_score = c, score
        return best if best_score >= 0.5 else None

    @staticmethod
    def _chunk_from_dict(d: dict) -> PolicyChunk:
        return PolicyChunk(
            chunk_id=d.get("chunk_id", ""), text=d.get("text", ""),
            clause_ref=d.get("clause_ref", ""), document_id=d.get("document_id", ""),
            title=d.get("title", ""), doc_type=d.get("doc_type", "policy"),
            product_line=d.get("product_line", ""), policy_id=d.get("policy_id", ""),
            version=d.get("version", "v1.0"),
            effective_date=d.get("effective_date", ""),
            supersedes=d.get("supersedes", ""))


def get_policy_store() -> PolicyKnowledgeStore:
    global _store
    try:
        return _store  # type: ignore[name-defined]
    except NameError:
        _store = PolicyKnowledgeStore()
        return _store
