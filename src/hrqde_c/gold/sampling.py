"""Stichprobenziehung mit Kalibrier-/Test-Split (AF-C.3.01, SA-C.3.06).

Die Zufallszuordnung passiert einmal bei der Ziehung und wird im Manifest
persistiert, damit spaetere Auswertungen dieselbe Teilung lesen.
"""

from __future__ import annotations

import json
import logging
import random
from datetime import date
from pathlib import Path

from hrqde_c.models import RawAdvertisement

log = logging.getLogger(__name__)


def draw(
    raws: list[RawAdvertisement], size: int, calib_ratio: float, seed: int
) -> dict[str, str]:
    """raw_id -> "kalibrier" | "test", deterministisch pro Seed."""
    rng = random.Random(seed)
    chosen = rng.sample(raws, min(size, len(raws)))
    n_calib = round(len(chosen) * calib_ratio)
    partition = {
        raw.id: "kalibrier" if i < n_calib else "test"
        for i, raw in enumerate(chosen)
    }
    log.info(
        "sampling: %d Anzeigen gezogen (%d Kalibrier, %d Test, Seed %d)",
        len(chosen), n_calib, len(chosen) - n_calib, seed,
    )
    return partition


def write_manifest(
    path: Path, partition: dict[str, str], seed: int, calib_ratio: float
) -> None:
    payload = {
        "drawn_at": date.today().isoformat(),
        "seed": seed,
        "calib_ratio": calib_ratio,
        "items": [{"raw_id": rid, "partition": part} for rid, part in partition.items()],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("sampling: Manifest -> %s", path)


def load_manifest(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {item["raw_id"]: item["partition"] for item in payload["items"]}
