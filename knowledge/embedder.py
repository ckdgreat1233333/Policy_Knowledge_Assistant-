"""Shared embedding wrapper.

Model choice: sentence-transformers/all-MiniLM-L6-v2
  - 384-dim vectors, fast CPU inference (fine for a clause corpus of ~100s),
  - strong retrieval quality on short technical text (insurance clauses),
  - embeddings are L2-normalized so FAISS IndexFlatL2 distance maps directly
    to cosine similarity:  sim = 1 - d^2/2.
"""
import numpy as np

_CACHED_MODEL = None
_CACHE_FAILED = False


def _get_model():
    global _CACHED_MODEL, _CACHE_FAILED
    if _CACHED_MODEL is not None:
        return _CACHED_MODEL
    if _CACHE_FAILED:
        return None
    try:
        from sentence_transformers import SentenceTransformer
        _CACHED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        return _CACHED_MODEL
    except ImportError:
        _CACHE_FAILED = True
        return None


class Embedder:

    def __init__(self):
        self.model = _get_model()
        self._available = self.model is not None

    def embed(self, texts: list[str]) -> np.ndarray:
        if not self._available or self.model is None:
            raise ImportError(
                "sentence-transformers is required for embedding. "
                "Install with: pip install sentence-transformers"
            )
        return self.model.encode(texts, convert_to_numpy=True,
                                 normalize_embeddings=True).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([query])
