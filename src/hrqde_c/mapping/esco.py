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

FULL_CONCEPTS_PATH = Path("data/esco/skills_full.json")
MINI_CONCEPTS_PATH = Path("data/esco/skills_mini.json")
DEFAULT_TOP_K = 3
ENCODER_ENV = "HRQDE_ENCODER"
CONCEPTS_ENV = "HRQDE_ESCO_CONCEPTS"


def _default_concepts_path() -> Path:
    env = os.environ.get(CONCEPTS_ENV)
    if env:
        return Path(env)
    # voller Index sobald gebaut, sonst der Mini-Bestand
    return FULL_CONCEPTS_PATH if FULL_CONCEPTS_PATH.exists() else MINI_CONCEPTS_PATH


DEFAULT_CONCEPTS_PATH = _default_concepts_path()

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
    # das Label, ueber das der Treffer zustande kam (pref oder alt, DE oder EN)
    label: str = ""


def _label_table(concepts: list[EscoConcept]) -> tuple[list[str], list[int]]:
    """Alle Labels flach, owners[i] zeigt auf das Konzept zu labels[i]."""
    labels: list[str] = []
    owners: list[int] = []
    for idx, concept in enumerate(concepts):
        for label in [concept.pref_label, *concept.alt_labels]:
            labels.append(label)
            owners.append(idx)
    return labels, owners


class EscoIndex:
    """Matcht auf Label-Ebene, dedupliziert die Treffer aufs Konzept."""

    def __init__(
        self,
        concepts: list[EscoConcept],
        labels: list[str],
        owners: list[int],
        embeddings: "torch.Tensor",
        model_id: str,
    ):
        self.concepts = concepts
        self.labels = labels
        self.owners = owners
        self.embeddings = embeddings
        self.model_id = model_id

    def match(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[Match]:
        import torch

        query_emb = encode([query], model_id=self.model_id, mode="query")[0]
        scores = torch.matmul(self.embeddings, query_emb)
        # mehr Labels ziehen als Konzepte gebraucht werden, ein Konzept kann
        # mit mehreren Labels vorne liegen
        k = min(max(top_k * 5, 20), len(self.labels))
        top_scores, top_idx = torch.topk(scores, k=k)

        best: dict[int, Match] = {}
        for score, label_idx in zip(top_scores, top_idx):
            owner = self.owners[label_idx]
            if owner not in best:
                best[owner] = Match(
                    concept=self.concepts[owner],
                    score=float(score),
                    label=self.labels[label_idx],
                )
            if len(best) >= top_k:
                break
        return sorted(best.values(), key=lambda m: m.score, reverse=True)


def _slug(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", model_id).strip("_")


def _cache_path(concepts_path: Path, model_id: str) -> Path:
    return concepts_path.parent / f"{concepts_path.stem}.{_slug(model_id)}.embeddings.pt"


def _load_from_cache(cache_path: Path, expected_labels: list[str], expected_model: str):
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
    if list(blob.get("labels", [])) != expected_labels:
        log.info("esco-index: Cache-Labels weichen ab, neu bauen")
        return None
    return blob["embeddings"]


def _save_cache(cache_path: Path, embeddings, labels: list[str], model_id: str) -> None:
    import torch

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_id": model_id, "labels": labels, "embeddings": embeddings}, cache_path
    )
    log.info("esco-index: Cache geschrieben nach %s", cache_path)


@lru_cache(maxsize=4)
def get_index(
    concepts_path: Path = DEFAULT_CONCEPTS_PATH,
    model_id: str = DEFAULT_MODEL_ID,
) -> EscoIndex:
    concepts = load_concepts(concepts_path)
    labels, owners = _label_table(concepts)
    cache_path = _cache_path(concepts_path, model_id)

    cached = _load_from_cache(cache_path, labels, model_id)
    if cached is not None:
        log.info(
            "esco-index: Cache-Hit (%d Konzepte, %d Labels, %s)",
            len(concepts), len(labels), model_id,
        )
        return EscoIndex(concepts, labels, owners, cached, model_id)

    log.info(
        "esco-index: encode %d Labels aus %d Konzepten mit %s",
        len(labels), len(concepts), model_id,
    )
    embeddings = encode(labels, model_id=model_id, mode="passage")
    _save_cache(cache_path, embeddings, labels, model_id)
    return EscoIndex(concepts, labels, owners, embeddings, model_id)
