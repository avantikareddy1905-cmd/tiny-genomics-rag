"""
Local RAG pipeline using ChromaDB + TinyLlama (Apache 2.0, fully open)
CPU-only, works on Apple Silicon or Linux.

Steps:
1. Embed and index a local text corpus with SentenceTransformer
2. Retrieve top-k relevant snippets for a user query
3. Summarize and generate an answer with citations using TinyLlama

Usage:
  python rag.py --corpus corpus.jsonl --query "What is known about APOE and Alzheimer’s disease?"
"""

import json
import argparse
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import chromadb
from chromadb.config import Settings

# ---------------- CONFIG ----------------
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # fully open model
TOP_K_DEFAULT = 3
MAX_SENTS_GEN = 8
# ----------------------------------------

# ---------------- LOAD CORPUS ----------------
def load_corpus(path: str) -> List[Dict[str, str]]:
    """Load the text corpus (JSONL file with {"id": "...", "text": "..."})"""
    corpus = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                corpus.append(json.loads(line))
    return corpus

# ---------------- HELPER FUNCTIONS ----------------
def count_sentences(text: str) -> int:
    return sum(ch in ".?!" for ch in text)

def trim_to_n_sentences(text: str, n: int) -> str:
    out, cnt = [], 0
    for ch in text:
        out.append(ch)
        if ch in ".?!":
            cnt += 1
            if cnt >= n:
                break
    return "".join(out).strip()

def build_context(snips: List[Dict]) -> str:
    """Join top-k snippets with source tags [S#]"""
    return "\n".join([f"[{s['id']}] {s['text']}" for s in snips])

# ---------------- RETRIEVAL ----------------
def build_chroma_collection(corpus: List[Dict[str, str]], emb_model: SentenceTransformer):
    """Create or reuse a ChromaDB collection"""
    client = chromadb.Client(Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection("corpus")

    if collection.count() == 0:
        texts = [c["text"] for c in corpus]
        ids = [c["id"] for c in corpus]
        emb = emb_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        collection.add(ids=ids, documents=texts, embeddings=emb.tolist())

    return collection

def retrieve(collection, query: str, k: int = TOP_K_DEFAULT):
    """Return top-k relevant text snippets"""
    results = collection.query(query_texts=[query], n_results=k)
    ids, docs = results["ids"][0], results["documents"][0]
    return [{"id": i, "text": d} for i, d in zip(ids, docs)]

# ---------------- GENERATION ----------------
def generate_answer(query: str, snips: List[Dict]) -> str:
    """Use TinyLlama to generate an 8-sentence scientific answer with citations"""
    if not snips:
        return f"Q: {query}\nA: No relevant evidence found.\n---"

    context = build_context(snips)

    # Chat-style prompt for TinyLlama
    system_msg = (
        "You are a biomedical research assistant. "
        "Answer user questions concisely using the provided EVIDENCE. "
        "Write in a scientific tone, no more than 8 sentences, and include inline citations like [S2][S3]."
    )

    user_msg = (
        f"QUESTION:\n{query}\n\n"
        f"EVIDENCE:\n{context}\n\n"
        "Provide your summarized answer based on the evidence below. Do not recite the evidence"
    )

    chat_prompt = f"<|system|>\n{system_msg}\n<|user|>\n{user_msg}\n<|assistant|>"

    # Initialize model
    generator = pipeline("text-generation", model=GEN_MODEL_NAME, torch_dtype="auto", device=-1)

    out = generator(
        chat_prompt,
        max_new_tokens=400,
        do_sample=False,

    )[0]["generated_text"].strip()

    # Clean up any leftover tokens
    out = out.replace("<|system|>", "").replace("<|user|>", "").replace("<|assistant|>", "").strip()

    # Limit to 8 sentences
    if count_sentences(out) > MAX_SENTS_GEN:
        out = trim_to_n_sentences(out, MAX_SENTS_GEN)

    # Add fallback citations if model forgot
    if "[" not in out:
        cites = "".join([f"[{s['id']}]" for s in snips])
        out = f"{out} {cites}"

    return f"Q: {query}\nA: {out}\n---"

# ---------------- MAIN CLI ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, help="Path to corpus.jsonl")
    parser.add_argument("--query", help="Single query string")
    parser.add_argument("--queries", help="Path to multiple queries.jsonl")
    parser.add_argument("--k", type=int, default=TOP_K_DEFAULT, help="Top-k snippets to retrieve")
    args = parser.parse_args()

    # Load and embed corpus
    corpus = load_corpus(args.corpus)
    emb_model = SentenceTransformer(EMB_MODEL_NAME)
    collection = build_chroma_collection(corpus, emb_model)

    # Single query
    if args.query:
        hits = retrieve(collection, args.query, args.k)
        print("Retrieved:", [h["id"] for h in hits])
        print(generate_answer(args.query, hits))
        return

    # Multiple queries
    if args.queries:
        with open(args.queries, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                q = json.loads(line)["q"]
                hits = retrieve(collection, q, args.k)
                print("Retrieved:", [h["id"] for h in hits])
                print(generate_answer(q, hits))
        return

    print("Please provide either --query or --queries")

# ---------------- RUN ----------------
if __name__ == "__main__":
    main()