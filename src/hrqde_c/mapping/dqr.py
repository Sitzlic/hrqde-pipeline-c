"""Berufsprofil-gebundene DQR-Niveau-Heuristik (RC-C.2.5).

Dreistufig: (1) ESCO-Occupation-Lookup ueber den Anzeigentitel, (2) Default
pro ISCO-08-Hauptgruppe, (3) Default-Niveau DQR4.

Abweichung von der Konzeption (dort: Mehrheitsvotum der extrahierten Spans
gegen die Essential-Skills eines Profils): Auf Echtdaten deckt kein einzelnes
Berufsprofil die Mehrheit der 30-60 Spans einer Anzeige ab (ein Profil hat
~20 Essential-Skills, generische Skills streuen ueber hunderte Profile), das
Votum liefert nur Gleichstaende. Der Anzeigentitel ist das verlaessliche
Berufsprofil-Signal: er wird gegen die ESCO-Occupation-Labels encodiert, der
aehnlichste Treffer oberhalb eines Schwellenwerts bestimmt das Profil. Bei
unsicherem Titel (kein Treffer ueber der Schwelle) faellt die Heuristik sauber
auf die naechste Stufe, statt falsch zuzuordnen.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from hrqde_c.mapping import esco
from hrqde_c.models import EscoOccupation

log = logging.getLogger(__name__)

OCCUPATIONS_ENV = "HRQDE_ESCO_OCCUPATIONS"
DEFAULT_OCCUPATIONS_PATH = Path("data/esco/occupations_full.json")

# Mindest-Kosinus fuer einen belastbaren Titel-Treffer, plus Mindestabstand
# zum Zweitplatzierten, damit zwei fast gleich gute Profile nicht raten
TITLE_THRESHOLD = 0.80
TITLE_MARGIN = 0.03

# DQR-Default pro ISCO-Hauptgruppe, Tabelle aus der Konzeption (RC-C.2.5)
ISCO_MAJOR_GROUP_DEFAULT: dict[int, str] = {
    1: "DQR7",  # Managers
    2: "DQR6",  # Professionals
    3: "DQR5",  # Technicians & Associates
    4: "DQR4",  # Clerical
    5: "DQR4",  # Service & Sales
    6: "DQR3",  # Skilled Agricultural
    7: "DQR3",  # Craft & Trades
    8: "DQR3",  # Plant & Machine Operators
    9: "DQR2",  # Elementary
}

# DQR4 = abgeschlossene Ausbildung, konservative Mitte
FALLBACK_LEVEL = "DQR4"

# Geschlechts- und Formzusaetze im Titel stoeren den Label-Match
_TITLE_NOISE = re.compile(
    r"\((?:[wmdfx]/?)+\)|/-?in\b|/-?innen\b|\bm/w/d\b|\bw/m/d\b",
    re.IGNORECASE,
)


@dataclass
class DqrResult:
    level: str
    occupation_uri: str | None
    # occupation-lookup | isco-default | fallback
    stage: str


def _occupations_path() -> Path:
    return Path(os.environ.get(OCCUPATIONS_ENV, DEFAULT_OCCUPATIONS_PATH))


def _normalize_title(title: str) -> str:
    return _TITLE_NOISE.sub(" ", title).strip(" -–,")


@lru_cache(maxsize=2)
def _load_occupations(path_str: str) -> tuple[EscoOccupation, ...]:
    path = Path(path_str)
    if not path.exists():
        log.info("dqr: %s fehlt, Occupation-Lookup inaktiv", path)
        return ()
    import json

    raw = json.loads(path.read_text(encoding="utf-8"))
    return tuple(EscoOccupation(**item) for item in raw)


@lru_cache(maxsize=2)
def _occupation_index(path_str: str, model_id: str):
    """(occupations, embeddings) mit Label-Cache analog zum ESCO-Skill-Index."""
    occupations = _load_occupations(path_str)
    if not occupations:
        return (), None
    labels = [o.pref_label for o in occupations]
    cache_path = esco._cache_path(Path(path_str), model_id)
    cached = esco._load_from_cache(cache_path, labels, model_id)
    if cached is not None:
        log.info("dqr: Occupation-Index Cache-Hit (%d Profile)", len(occupations))
        return occupations, cached
    log.info("dqr: encode %d Occupation-Labels mit %s", len(labels), model_id)
    embeddings = esco.encode(labels, model_id=model_id, mode="passage")
    esco._save_cache(cache_path, embeddings, labels, model_id)
    return occupations, embeddings


def lookup(
    title: str,
    occupations: tuple[EscoOccupation, ...],
    embeddings,
    model_id: str = esco.DEFAULT_MODEL_ID,
) -> EscoOccupation | None:
    """Aehnlichstes ESCO-Berufsprofil zum Anzeigentitel oberhalb der Schwelle."""
    if not occupations or embeddings is None:
        return None
    import torch

    query = _normalize_title(title)
    if not query:
        return None
    q = esco.encode([query], model_id=model_id, mode="query")[0]
    scores = torch.matmul(embeddings, q)
    k = min(2, len(occupations))
    top = torch.topk(scores, k)
    best_score = float(top.values[0])
    if best_score < TITLE_THRESHOLD:
        return None
    if k > 1 and best_score - float(top.values[1]) < TITLE_MARGIN:
        return None
    return occupations[int(top.indices[0])]


def determine(title: str, isco_group: int | None) -> DqrResult:
    """Dreistufige Heuristik: Titel-Occupation-Lookup -> ISCO-Hauptgruppen-Default
    -> Default-Niveau DQR4."""
    occupations, embeddings = _occupation_index(str(_occupations_path()), esco.DEFAULT_MODEL_ID)
    occ = lookup(title, occupations, embeddings)
    if occ is not None:
        level = ISCO_MAJOR_GROUP_DEFAULT.get(occ.isco_major_group, FALLBACK_LEVEL)
        log.info("dqr: Titel-Lookup %r -> %s (%s), Niveau %s",
                 title, occ.pref_label, occ.code, level)
        return DqrResult(level=level, occupation_uri=occ.uri, stage="occupation-lookup")
    if isco_group is not None:
        level = ISCO_MAJOR_GROUP_DEFAULT.get(isco_group)
        if level is not None:
            return DqrResult(level=level, occupation_uri=None, stage="isco-default")
        log.warning("dqr: unbekannte ISCO-Hauptgruppe %r, nutze Fallback", isco_group)
    return DqrResult(level=FALLBACK_LEVEL, occupation_uri=None, stage="fallback")
