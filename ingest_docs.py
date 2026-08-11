"""
ingest_docs.py — feed real provider PDFs into the RAG knowledge store.

Usage:
    1. Drop PDF files into ./knowledge_docs/
    2. Run:  python ingest_docs.py

Requires pdftotext (poppler):  brew install poppler
"""

import os
import re
import subprocess
import chromadb
from openai import OpenAI
from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)
EMBED_MODEL = "text-embedding-3-small"
DOCS_DIR = "./knowledge_docs"
CHUNK_WORDS = 180
CHUNK_OVERLAP_WORDS = 30


def extract_text(pdf_path):
    result = subprocess.run(
        ["pdftotext", "-layout", pdf_path, "-"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"pdftotext failed on {pdf_path}: {result.stderr}")
    return result.stdout


def clean_text(text):
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def chunk_text(text, source):
    words = text.split()
    chunks = []
    i = 0
    idx = 0
    while i < len(words):
        piece = words[i:i + CHUNK_WORDS]
        if len(piece) < 25:
            break
        chunks.append({
            "id": f"{source}::chunk-{idx}",
            "text": " ".join(piece),
            "tags": f"source:{source}",
        })
        idx += 1
        i += CHUNK_WORDS - CHUNK_OVERLAP_WORDS
    return chunks


def embed(texts):
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def get_collection():
    db = chromadb.PersistentClient(path="./chroma_store")
    return db.get_or_create_collection("protection_knowledge")


def ingest_all():
    if not os.path.isdir(DOCS_DIR):
        print(f"No {DOCS_DIR} folder found. Create it and drop PDFs in.")
        return
    pdfs = [f for f in os.listdir(DOCS_DIR) if f.lower().endswith(".pdf")]
    if not pdfs:
        print(f"No PDFs found in {DOCS_DIR}/")
        return
    col = get_collection()
    total = 0
    for filename in pdfs:
        path = os.path.join(DOCS_DIR, filename)
        print(f"\nReading {filename} ...")
        text = clean_text(extract_text(path))
        if len(text) < 200:
            print("  ⚠ Very little text — may be scanned/image-based. Skipping.")
            continue
        chunks = chunk_text(text, source=filename)
        if not chunks:
            print("  ⚠ No usable chunks. Skipping.")
            continue
        texts = [c["text"] for c in chunks]
        ids = [c["id"] for c in chunks]
        metadatas = [{"tags": c["tags"]} for c in chunks]
        vectors = embed(texts)
        col.upsert(ids=ids, documents=texts, embeddings=vectors, metadatas=metadatas)
        print(f"  ✓ Added {len(chunks)} chunks from {filename}")
        total += len(chunks)
    print(f"\nDone. {total} chunks ingested from {len(pdfs)} document(s).")


if __name__ == "__main__":
    ingest_all()