"""
rag.py — Retrieval-Augmented Generation for the protection co-pilot.

Builds a local Chroma vector store from a protection-knowledge corpus and
retrieves the most relevant passages for a given client situation — so the
model can be given ONLY the knowledge that matters, instead of pasting the
whole knowledge base into every prompt.

Run directly to build the store and test retrieval:
    python rag.py
"""

import os
import chromadb
from openai import OpenAI
from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)
EMBED_MODEL = "text-embedding-3-small"

KNOWLEDGE_CHUNKS = [
    {"id": "meth-income-foundation",
     "text": "Income is the foundation of protection advice. Assess need based on household income, not mortgage payments — they are not linked. Everything the client has depends on their ability to earn.",
     "tags": "methodology income"},
    {"id": "meth-advice-not-sale",
     "text": "Protection is advice, not a sale. The client came for advice. The goal is not the mortgage, it is the home — and protection secures the ability to keep it. Frame the adviser as an expert working for the client.",
     "tags": "methodology philosophy"},
    {"id": "meth-income-stress-test",
     "text": "Income stress test: if the client's income stopped, how many months of savings before they run out? Could they clear the debt? Who depends on that income? This makes the need concrete.",
     "tags": "methodology income-protection"},
    {"id": "prod-ip-claim-length",
     "text": "The average income protection claim lasts around 7 years 8 months. This is why full-term income protection matters — short-term cover can run out long before the client recovers or returns to work.",
     "tags": "income-protection statistic"},
    {"id": "prod-ip-claim-causes",
     "text": "Over half of income protection claims (around 56%) are for mental health or musculoskeletal conditions. Critical illness cover typically would NOT pay out for these — a key reason IP and CIC are not interchangeable.",
     "tags": "income-protection critical-illness statistic"},
    {"id": "prod-cic-mortgage",
     "text": "Less than 1% of decreasing critical illness cover ends up paying off the mortgage. Decreasing CIC is often mismatched to the actual need; consider whether level cover or income protection better fits.",
     "tags": "critical-illness statistic"},
    {"id": "prod-fib",
     "text": "Family Income Benefit pays a regular monthly income to the family for the remainder of the term after death, rather than a lump sum. Often cheaper than level term life cover and a better fit for replacing a breadwinner's income for dependants.",
     "tags": "life-cover family-income-benefit dependants"},
    {"id": "prod-underwriting-future",
     "text": "Standard (ordinary) underwriting terms protect against future uninsurability. Securing cover while healthy matters because health can change and make cover more expensive or unavailable later.",
     "tags": "underwriting health"},
    {"id": "occ-own-occupation",
     "text": "For clients whose income depends on a specific physical capability (surgeons' hands, manual trades), an 'own occupation' income protection definition matters — it pays if they cannot do THEIR job, not just any job.",
     "tags": "occupation income-protection surgeon manual"},
    {"id": "occ-self-employed",
     "text": "Self-employed clients usually have no employer sick pay — no safety net if they cannot work. This makes income protection especially relevant, and the deferred period should reflect how long their savings could realistically last.",
     "tags": "occupation self-employed income-protection"},
    {"id": "occ-employer-benefits",
     "text": "Employer benefits (death-in-service, sick pay) are conditional: tied to remaining with that employer, can be cut, and are sometimes only underwritten at point of claim. Treat them as a bonus, not the foundation of a protection plan.",
     "tags": "employer-benefits death-in-service"},
    {"id": "val-proportion",
     "text": "Frame cost as a small proportion of what it safeguards: the client is insuring their income and their family's security. Ask what it would be worth to protect that income, rather than leading with price. Never quote a premium — the adviser sources real prices.",
     "tags": "value-framing cost"},
]


def get_collection(reset=False):
    db = chromadb.CloudClient(
    api_key=os.getenv("CHROMA_API_KEY"),
    tenant=os.getenv("CHROMA_TENANT"),
    database=os.getenv("CHROMA_DATABASE"),
)
    if reset:
        try:
            db.delete_collection("protection_knowledge")
        except Exception:
            pass
    return db.get_or_create_collection("protection_knowledge")


def embed(texts):
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def build_store():
    col = get_collection(reset=True)
    texts = [c["text"] for c in KNOWLEDGE_CHUNKS]
    ids = [c["id"] for c in KNOWLEDGE_CHUNKS]
    metadatas = [{"tags": c["tags"]} for c in KNOWLEDGE_CHUNKS]
    vectors = embed(texts)
    col.add(ids=ids, documents=texts, embeddings=vectors, metadatas=metadatas)
    print(f"Built store with {len(ids)} knowledge chunks.")
    return col


def retrieve(situation, k=4):
    col = get_collection()
    qvec = embed([situation])[0]
    res = col.query(query_embeddings=[qvec], n_results=k)
    docs = res["documents"][0]
    ids = res["ids"][0]
    return list(zip(ids, docs))


if __name__ == "__main__":
    build_store()
    tests = [
        "Self-employed plumber, 45, worried about injury stopping his income, two kids.",
        "Surgeon whose income depends on operating, wants to protect his family.",
        "Client has death-in-service through his employer and thinks that's enough.",
    ]
    for t in tests:
        print("\n" + "=" * 70)
        print("SITUATION:", t)
        print("-" * 70)
        for cid, doc in retrieve(t, k=3):
            print(f"  [{cid}]  {doc[:110]}...")