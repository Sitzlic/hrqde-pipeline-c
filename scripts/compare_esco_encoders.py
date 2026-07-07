"""Vergleicht die drei Encoder-Kandidaten auf den Test-Queries gegen den Mini-Index.

Aufruf: python scripts/compare_esco_encoders.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer


CANDIDATES = [
    ("baseline SBERT", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", None, None),
    ("TH Luebeck ESCO", "isy-thl/multilingual-e5-base-course-skill-tuned", "query: ", "passage: "),
    ("TechWolf ConTeXT", "TechWolf/ConTeXT-Skill-Extraction-base", None, None),
]

QUERIES = [
    "Java-Kenntnissen",
    "Projektmanagement",
    "Teamfaehigkeit",
    "agilen Methoden",
    "Buchhaltung",
    "arbeitest",
]


def load_concepts():
    payload = json.loads(Path("data/esco/skills_mini.json").read_text())
    return [(item["uri"], item["pref_label"]) for item in payload]


def encode(model: SentenceTransformer, texts: list[str], prefix: str | None) -> torch.Tensor:
    if prefix:
        texts = [prefix + t for t in texts]
    return model.encode(
        texts, normalize_embeddings=True, convert_to_tensor=True, device="cpu"
    ).cpu()


def run() -> None:
    concepts = load_concepts()
    labels = [label for _, label in concepts]

    for name, model_id, query_prefix, passage_prefix in CANDIDATES:
        print(f"\n{'=' * 70}\n{name}: {model_id}\n{'=' * 70}")
        model = SentenceTransformer(model_id)
        concept_emb = encode(model, labels, passage_prefix)
        for query in QUERIES:
            query_emb = encode(model, [query], query_prefix)[0]
            scores = torch.matmul(concept_emb, query_emb).tolist()
            ranked = sorted(zip(labels, scores), key=lambda x: -x[1])
            top1_label, top1_score = ranked[0]
            top2_score = ranked[1][1]
            margin = top1_score - top2_score
            print(f"  {query:22s} -> {top1_label:32s}  "
                  f"top1={top1_score:.3f}  margin={margin:+.3f}")


if __name__ == "__main__":
    run()
