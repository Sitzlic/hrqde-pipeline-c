"""Holt alle ESCO-Skills ueber die oeffentliche API und baut skills_full.json.

Labels DE+EN (SA-C.2.04): pref_label ist das deutsche preferredLabel,
alt_labels sammelt das englische preferredLabel plus alle alternativeLabel
in beiden Sprachen. Format kompatibel zu models.EscoConcept.

Der /search-Endpoint deckelt bei 200 Treffern, deshalb laeuft das hier
ueber /resource/skill mit dem member-skills-Schema. Dort ist "offset"
eine Seitennummer. Einzelne Seiten quittiert der Server dauerhaft mit
HTTP 500 (kaputtes Konzept im Bereich), die werden bisektiert und nur
das defekte Konzept uebersprungen.

Aufruf: python scripts/fetch_esco_skills.py [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BULK_URL = "https://ec.europa.eu/esco/api/resource/skill"
MEMBER_SKILLS_SCHEME = "http://data.europa.eu/esco/concept-scheme/member-skills"
# Seitenbreiten fuer die Bisektion, jede teilt die vorherige ganzzahlig
WIDTHS = [200, 100, 50, 25, 5, 1]
RETRIES = 3
# feste ESCO-Version, sonst liefert die API die jeweils aktuelle (Kap. 2: v1.2.1)
ESCO_VERSION = "v1.2.1"


def _get(limit: int, page_no: int) -> dict | None:
    """Eine Seite. None bei dauerhaftem Serverfehler, Exception bei Netzproblemen."""
    params = {
        "isInScheme": MEMBER_SKILLS_SCHEME,
        "language": "de",
        "selectedVersion": ESCO_VERSION,
        "limit": limit,
        "offset": page_no,
    }
    url = f"{BULK_URL}?{urlencode(params)}"
    last_exc: Exception | None = None
    for attempt in range(RETRIES):
        try:
            req = Request(url, headers={"Accept": "application/json"})
            with urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            if exc.code >= 500:
                # Serverfehler bleiben auf manchen Seiten bestehen -> bisektieren
                last_exc = exc
                time.sleep(1)
                continue
            raise
        except Exception as exc:  # noqa: BLE001 - Netz-Huster, nochmal
            last_exc = exc
            time.sleep(2 * (attempt + 1))
    if isinstance(last_exc, HTTPError):
        return None
    raise RuntimeError(f"ESCO-API nach {RETRIES} Versuchen nicht erreichbar: {last_exc}")


def _dedupe(labels: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for label in labels:
        key = label.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(label.strip())
    return out


def to_concept(hit: dict) -> dict | None:
    pref = hit.get("preferredLabel") or {}
    pref_de, pref_en = pref.get("de"), pref.get("en")
    if not pref_de and not pref_en:
        return None
    alt = hit.get("alternativeLabel") or {}
    alt_labels = _dedupe(
        ([pref_en] if pref_de and pref_en else [])
        + list(alt.get("de") or [])
        + list(alt.get("en") or [])
    )
    desc = hit.get("description") or {}
    definition = None
    for lang in ("de", "en"):
        entry = desc.get(lang)
        if isinstance(entry, dict):
            definition = entry.get("literal")
        elif isinstance(entry, str):
            definition = entry
        if definition:
            break
    return {
        "uri": hit["uri"],
        "pref_label": pref_de or pref_en,
        "alt_labels": alt_labels,
        "definition": definition,
    }


def collect(width_idx: int, page_no: int, out: list[dict], skipped: list[int]) -> None:
    width = WIDTHS[width_idx]
    page = _get(width, page_no)
    if page is not None:
        out.extend(page["_embedded"].values())
        return
    if width == 1:
        # bei Breite 1 ist die Seitennummer der Element-Index
        print(f"\nKonzept an Position {page_no} kaputt (HTTP 500), uebersprungen")
        skipped.append(page_no)
        return
    next_width = WIDTHS[width_idx + 1]
    first = page_no * width // next_width
    last = (page_no * width + width) // next_width
    for sub in range(first, last):
        collect(width_idx + 1, sub, out, skipped)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/esco/skills_full.json"))
    args = parser.parse_args()

    first_page = _get(WIDTHS[0], 0)
    if first_page is None:
        raise RuntimeError("schon die erste Seite liefert HTTP 500")
    total = first_page["total"]
    hits: list[dict] = list(first_page["_embedded"].values())
    skipped: list[int] = []

    page_no = 1
    while len(hits) + len(skipped) < total:
        collect(0, page_no, hits, skipped)
        page_no += 1
        print(f"\r{len(hits)}/{total}", end="", flush=True)
        time.sleep(0.3)

    print()
    concepts = [c for c in (to_concept(h) for h in hits) if c]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(concepts, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    label_count = sum(1 + len(c["alt_labels"]) for c in concepts)
    print(f"{len(concepts)} Konzepte, {label_count} Labels -> {args.out}")
    if skipped:
        print(f"{len(skipped)} kaputte Konzepte uebersprungen (Positionen {skipped})")


if __name__ == "__main__":
    main()
