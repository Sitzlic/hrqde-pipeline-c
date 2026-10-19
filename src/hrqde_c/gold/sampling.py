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


def draw_stratified(
    strata: dict[str, list[RawAdvertisement]], size: int, calib_ratio: float, seed: int
) -> tuple[dict[str, str], dict[str, str]]:
    """Gleich viele Anzeigen je Schicht (Berufsfeld), innerhalb jeder Schicht
    haelftig Kalibrier-/Testteil. -> (raw_id -> partition, raw_id -> stratum)"""
    if size % len(strata):
        raise ValueError(f"Stichprobe {size} nicht gleichmaessig auf {len(strata)} Schichten teilbar")
    per_stratum = size // len(strata)
    rng = random.Random(seed)
    partition: dict[str, str] = {}
    stratum_of: dict[str, str] = {}
    for name, raws in strata.items():
        if len(raws) < per_stratum:
            raise ValueError(
                f"Schicht {name}: nur {len(raws)} Anzeigen, {per_stratum} benoetigt (UCon3-UCas1)"
            )
        chosen = rng.sample(raws, per_stratum)
        n_calib = round(len(chosen) * calib_ratio)
        for i, raw in enumerate(chosen):
            partition[raw.id] = "kalibrier" if i < n_calib else "test"
            stratum_of[raw.id] = name
        log.info("sampling: Schicht %s: %d gezogen (%d Kalibrier, %d Test)",
                 name, len(chosen), n_calib, len(chosen) - n_calib)
    log.info("sampling: %d Anzeigen in %d Schichten, Seed %d", len(partition), len(strata), seed)
    return partition, stratum_of


def write_manifest(
    path: Path, partition: dict[str, str], seed: int, calib_ratio: float,
    stratum_of: dict[str, str] | None = None,
) -> None:
    items = []
    for rid, part in partition.items():
        item = {"raw_id": rid, "partition": part}
        if stratum_of and rid in stratum_of:
            item["stratum"] = stratum_of[rid]
        items.append(item)
    payload = {
        "drawn_at": date.today().isoformat(),
        "seed": seed,
        "calib_ratio": calib_ratio,
        "items": items,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("sampling: Manifest -> %s", path)


def load_manifest(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {item["raw_id"]: item["partition"] for item in payload["items"]}
