"""Embed a query and fetch the closest chunks from ChromaDB."""
from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer

import config


@lru_cache(maxsize=1)
def get_embedder():
    return SentenceTransformer(config.EMBED_MODEL)


@lru_cache(maxsize=1)
def get_client():
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def index_size() -> int:
    try:
        return get_client().get_collection(config.COLLECTION_NAME).count()
    except Exception:
        return 0


def retrieve(query: str, k: int = config.TOP_K):
    """Return a list of {text, source, page, score} dicts (best first)."""
    collection = get_client().get_collection(config.COLLECTION_NAME)
    q_emb = get_embedder().encode([query], normalize_embeddings=True).tolist()
    res = collection.query(query_embeddings=q_emb, n_results=k)
    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append({
            "text": doc,
            "source": meta.get("source", "?"),
            "page": meta.get("page", "?"),
            "score": round(1 - dist, 3),  # cosine similarity
        })
    return hits
