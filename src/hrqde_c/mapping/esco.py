from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from hrqde_c.models import EscoConcept

if TYPE_CHECKING:
    import torch
    from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

DEFAULT_CONCEPTS_PATH = Path("data/esco/skills_mini.json")
DEFAULT_TOP_K = 3
ENCODER_ENV = "HRQDE_ENCODER"

# E5-Modelle wollen query:/passage:-Prefixes, alle anderen laufen ohne
ENCODER_PROFILES: dict[str, dict[str, str]] = {
    "TechWolf/ConTeXT-Skill-Extraction-base": {},
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": {},
    "isy-thl/multilingual-e5-base-course-skill-tuned": {
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
}

DEFAULT_MODEL_ID = os.environ.get(
    ENCODER_ENV, "TechWolf/ConTeXT-Skill-Extraction-base"
)


def load_concepts(path: Path = DEFAULT_CONCEPTS_PATH) -> list[EscoConcept]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [EscoConcept(**item) for item in raw]


@lru_cache(maxsize=4)
def _load_encoder(model_id: str) -> "SentenceTransformer":
    from sentence_transformers import SentenceTransformer

    log.info("esco-encoder: lade %s", model_id)
    return SentenceTransformer(model_id)


def _prefix_for(model_id: str, mode: Literal["query", "passage"]) -> str:
    profile = ENCODER_PROFILES.get(model_id, {})
    return profile.get(f"{mode}_prefix", "")


def encode(
    texts: list[str],
    model_id: str = DEFAULT_MODEL_ID,
    mode: Literal["query", "passage"] = "passage",
) -> "torch.Tensor":
    prefix = _prefix_for(model_id, mode)
    if prefix:
        texts = [prefix + t for t in texts]

    encoder = _load_encoder(model_id)
    embeddings = encoder.encode(
        texts,
        normalize_embeddings=True,
        convert_to_tensor=True,
        device="cpu",
        show_progress_bar=False,
    )
    return embeddings.cpu()


@dataclass
class Match:
    concept: EscoConcept
    score: float


class EscoIndex:
    def __init__(self, concepts: list[EscoConcept], embeddings: "torch.Tensor", model_id: str):
        self.concepts = concepts
        self.embeddings = embeddings
        self.model_id = model_id

    def match(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[Match]:
        import torch

        query_emb = encode([query], model_id=self.model_id, mode="query")[0]
        scores = torch.matmul(self.embeddings, query_emb)
        top_scores, top_idx = torch.topk(scores, k=min(top_k, len(self.concepts)))
        return [Match(concept=self.concepts[i], score=float(s))
                for s, i in zip(top_scores, top_idx)]


def _slug(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", model_id).strip("_")


def _cache_path(concepts_path: Path, model_id: str) -> Path:
    return concepts_path.parent / f"{concepts_path.stem}.{_slug(model_id)}.embeddings.pt"


def _load_from_cache(cache_path: Path, expected_uris: list[str], expected_model: str):
    import torch

    if not cache_path.exists():
        return None
    try:
        blob = torch.load(cache_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        log.warning("esco-index: Cache %s nicht lesbar (%s)", cache_path, exc)
        return None
    if blob.get("model_id") != expected_model:
        return None
    if list(blob.get("uris", [])) != expected_uris:
        log.info("esco-index: Cache-Konzepte weichen ab, neu bauen")
        return None
    return blob["embeddings"]


def _save_cache(cache_path: Path, embeddings, uris: list[str], model_id: str) -> None:
    import torch

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_id": model_id, "uris": uris, "embeddings": embeddings}, cache_path)
    log.info("esco-index: Cache geschrieben nach %s", cache_path)


@lru_cache(maxsize=4)
def get_index(
    concepts_path: Path = DEFAULT_CONCEPTS_PATH,
    model_id: str = DEFAULT_MODEL_ID,
) -> EscoIndex:
    concepts = load_concepts(concepts_path)
    uris = [c.uri for c in concepts]
    cache_path = _cache_path(concepts_path, model_id)

    cached = _load_from_cache(cache_path, uris, model_id)
    if cached is not None:
        log.info("esco-index: Cache-Hit (%d Konzepte, %s)", len(concepts), model_id)
        return EscoIndex(concepts, cached, model_id)

    log.info("esco-index: encode %d Konzepte mit %s", len(concepts), model_id)
    embeddings = encode(
        [c.pref_label for c in concepts], model_id=model_id, mode="passage"
    )
    _save_cache(cache_path, embeddings, uris, model_id)
    return EscoIndex(concepts, embeddings, model_id)
