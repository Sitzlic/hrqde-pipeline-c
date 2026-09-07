"""Holt alle ESCO-Occupations fuer die DQR-Heuristik (RC-C.2.5 Primaerweg).

Pro Berufsprofil: URI, Labels DE+EN, ESCO-Code (ISCO-08-basiert, erste
Ziffer = Hauptgruppe) und die URIs der essentiellen Skills. Gleicher
Bulk-Endpoint wie bei den Skills, offset ist eine Seitennummer.

Aufruf: python scripts/fetch_esco_occupations.py [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BULK_URL = "https://ec.europa.eu/esco/api/resource/occupation"
SCHEME = "http://data.europa.eu/esco/concept-scheme/member-occupations"
# Seitenbreiten fuer die Bisektion um serverseitig kaputte Konzepte herum
WIDTHS = [200, 100, 50, 25, 5, 1]
RETRIES = 3


def _get(limit: int, page_no: int) -> dict | None:
    """None bei dauerhaftem Serverfehler (kaputte Seite), Exception bei Netz."""
    params = {"isInScheme": SCHEME, "language": "de",
              "limit": limit, "offset": page_no}
    url = f"{BULK_URL}?{urlencode(params)}"
    last_exc = None
    for attempt in range(RETRIES):
        try:
            req = Request(url, headers={"Accept": "application/json"})
            with urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            if exc.code >= 500:
                last_exc = exc
                time.sleep(1)
                continue
            raise
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(2 * (attempt + 1))
    if isinstance(last_exc, HTTPError):
        return None
    raise RuntimeError(f"ESCO-API nicht erreichbar: {last_exc}")


def collect(width_idx: int, page_no: int, out: list[dict], skipped: list[int]) -> None:
    width = WIDTHS[width_idx]
    page = _get(width, page_no)
    if page is not None:
        out.extend(page["_embedded"].values())
        return
    if width == 1:
        print(f"\nOccupation an Position {page_no} kaputt (HTTP 500), uebersprungen")
        skipped.append(page_no)
        return
    next_width = WIDTHS[width_idx + 1]
    for sub in range(page_no * width // next_width, (page_no * width + width) // next_width):
        collect(width_idx + 1, sub, out, skipped)


def to_occupation(item: dict) -> dict | None:
    pref = item.get("preferredLabel") or {}
    pref_de, pref_en = pref.get("de"), pref.get("en")
    code = item.get("code")
    if not code or (not pref_de and not pref_en):
        return None
    links = item.get("_links") or {}
    essential = [
        e["uri"] for e in links.get("hasEssentialSkill") or [] if e.get("uri")
    ]
    return {
        "uri": item["uri"],
        "pref_label": pref_de or pref_en,
        "code": code,
        "isco_major_group": int(code[0]),
        "essential_skills": essential,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/esco/occupations_full.json"))
    args = parser.parse_args()

    first = _get(WIDTHS[0], 0)
    if first is None:
        raise RuntimeError("schon die erste Seite liefert HTTP 500")
    total = first["total"]
    items: list[dict] = list(first["_embedded"].values())
    skipped: list[int] = []
    page_no = 1
    while len(items) + len(skipped) < total:
        collect(0, page_no, items, skipped)
        page_no += 1
        print(f"\r{len(items)}/{total}", end="", flush=True)
        time.sleep(0.3)

    print()
    occupations = [o for o in (to_occupation(i) for i in items) if o]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(occupations, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    ess = sum(len(o["essential_skills"]) for o in occupations)
    print(f"{len(occupations)} Occupations, {ess} Essential-Skill-Kanten -> {args.out}")
    if skipped:
        print(f"{len(skipped)} kaputte Occupations uebersprungen (Positionen {skipped})")


if __name__ == "__main__":
    main()
