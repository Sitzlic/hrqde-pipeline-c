from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from hrqde_c.models import EscoOccupation

log = logging.getLogger(__name__)

OCCUPATIONS_ENV = "HRQDE_ESCO_OCCUPATIONS"
DEFAULT_OCCUPATIONS_PATH = Path("data/esco/occupations_full.json")

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


@dataclass
class DqrResult:
    level: str
    occupation_uri: str | None
    # occupation-lookup | isco-default | fallback
    stage: str


def _occupations_path() -> Path:
    return Path(os.environ.get(OCCUPATIONS_ENV, DEFAULT_OCCUPATIONS_PATH))


@lru_cache(maxsize=2)
def _load_occupations(path_str: str) -> tuple[EscoOccupation, ...]:
    path = Path(path_str)
    if not path.exists():
        log.info("dqr: %s fehlt, Occupation-Lookup inaktiv", path)
        return ()
    raw = json.loads(path.read_text(encoding="utf-8"))
    return tuple(EscoOccupation(**item) for item in raw)


def lookup(accepted_uris: list[str], occupations: tuple[EscoOccupation, ...]) -> EscoOccupation | None:
    """Mehrheitsvotum nach Konzeption: die Occupation gewinnt, deren
    Essential-Skills die Mehrheit der akzeptierten Konzepte einer Anzeige
    abdecken. Kein alleiniger Spitzenreiter oder keine Mehrheit -> None."""
    concepts = set(accepted_uris)
    if not concepts or not occupations:
        return None
    best: EscoOccupation | None = None
    best_votes = 0
    tied = False
    for occ in occupations:
        votes = len(concepts.intersection(occ.essential_skills))
        if votes > best_votes:
            best, best_votes, tied = occ, votes, False
        elif votes == best_votes and votes > 0:
            tied = True
    if best is None or tied:
        return None
    if best_votes * 2 <= len(concepts):
        return None
    return best


def determine(accepted_uris: list[str], isco_group: int | None) -> DqrResult:
    """Dreistufige Heuristik: Occupation-Lookup -> ISCO-Hauptgruppen-Default
    -> Default-Niveau DQR4."""
    occ = lookup(accepted_uris, _load_occupations(str(_occupations_path())))
    if occ is not None:
        level = ISCO_MAJOR_GROUP_DEFAULT.get(occ.isco_major_group, FALLBACK_LEVEL)
        log.info("dqr: Occupation-Lookup -> %s (%s), Niveau %s",
                 occ.pref_label, occ.code, level)
        return DqrResult(level=level, occupation_uri=occ.uri, stage="occupation-lookup")
    if isco_group is not None:
        level = ISCO_MAJOR_GROUP_DEFAULT.get(isco_group)
        if level is not None:
            return DqrResult(level=level, occupation_uri=None, stage="isco-default")
        log.warning("dqr: unbekannte ISCO-Hauptgruppe %r, nutze Fallback", isco_group)
    return DqrResult(level=FALLBACK_LEVEL, occupation_uri=None, stage="fallback")
