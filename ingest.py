"""Indexing: read docs in data/, chunk, embed, store in ChromaDB.

Use from the Streamlit sidebar (Build index button) or from the terminal:
    python ingest.py
Re-run whenever documents (or EMBED_MODEL) change; it rebuilds the index.
"""
from pathlib import Path

from pypdf import PdfReader

import config
from retriever import get_client, get_embedder

SUPPORTED = (".pdf", ".txt", ".md")


def list_documents():
    config.DATA_DIR.mkdir(exist_ok=True)
    return [p for p in sorted(config.DATA_DIR.glob("**/*")) if p.suffix.lower() in SUPPORTED]


def read_file(path: Path):
    """Yield (page_number, text) for a PDF/TXT/MD file."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        for i, page in enumerate(reader.pages, start=1):
            yield i, page.extract_text() or ""
    elif suffix in (".txt", ".md"):
        yield 1, path.read_text(encoding="utf-8", errors="ignore")


def chunk_words(text: str, size: int, overlap: int):
    words = text.split()
    if not words:
        return
    step = max(size - overlap, 1)
    for start in range(0, len(words), step):
        piece = words[start:start + size]
        if piece:
            yield " ".join(piece)
        if start + size >= len(words):
            break


def build_index(show_progress: bool = False):
    """Rebuild the index from data/. Returns (n_chunks, n_files).

    Raises ValueError with a readable message if nothing can be indexed.
    """
    files = list_documents()
    if not files:
        raise ValueError(f"No documents found in {config.DATA_DIR}. Add PDF/TXT/MD files first.")

    ids, docs, metas = [], [], []
    for path in files:
        for page, text in read_file(path):
            for i, chunk in enumerate(chunk_words(text, config.CHUNK_WORDS, config.CHUNK_OVERLAP)):
                ids.append(f"{path.name}-p{page}-c{i}")
                docs.append(chunk)
                metas.append({"source": path.name, "page": page})

    if not docs:
        raise ValueError("No extractable text found (scanned PDFs need OCR).")

    embeddings = get_embedder().encode(
        docs, normalize_embeddings=True, show_progress_bar=show_progress
    ).tolist()

    client = get_client()
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    batch = 500
    for i in range(0, len(docs), batch):
        collection.add(
            ids=ids[i:i + batch],
            documents=docs[i:i + batch],
            embeddings=embeddings[i:i + batch],
            metadatas=metas[i:i + batch],
        )
    return len(docs), len(files)


def main():
    try:
        n_chunks, n_files = build_index(show_progress=True)
    except ValueError as e:
        print(e)
        return
    print(f"Done. Indexed {n_chunks} chunks from {n_files} file(s).")


if __name__ == "__main__":
    main()