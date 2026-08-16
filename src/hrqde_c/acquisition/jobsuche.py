"""Anbindung der BA-Jobsuche-API als Datenquelle (RC-C.1.2).

Freie API der Bundesagentur fuer Arbeit. Die Suche liefert Treffer-Metadaten,
erst der Detail-Aufruf den Beschreibungs-Volltext. Anzeigen ohne Volltext
(extern gehostete Inserate) werden uebersprungen.

Achtung: der Dienst blockt Anfragen aus dem Ausland, Aufrufe brauchen
eine deutsche IP (oder VPN).
"""

from __future__ import annotations

import base64
import json
import logging
import re
import time
from datetime import date
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from hrqde_c.models import RawAdvertisement

log = logging.getLogger(__name__)

BASE_URL = "https://rest.arbeitsagentur.de/jobboerse/jobsuche-service/pc/v4"
# oeffentlicher Key der freien API (bund.dev)
API_KEY = "jobboerse-jobsuche"
PLATFORM = "ba_jobsuche"
DETAIL_PAGE = "https://www.arbeitsagentur.de/jobsuche/jobdetail/{refnr}"
REQUEST_DELAY = 0.5


def _get(url: str) -> dict:
    req = Request(url, headers={
        "X-API-Key": API_KEY,
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0",
    })
    with urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def search(was: str, wo: str | None = None, size: int = 25, page: int = 1) -> list[dict]:
    params: dict = {"was": was, "size": size, "page": page}
    if wo:
        params["wo"] = wo
    payload = _get(f"{BASE_URL}/jobs?{urlencode(params)}")
    hits = payload.get("stellenangebote") or []
    log.info("jobsuche: %r Seite %d -> %d Treffer", was, page, len(hits))
    return hits


def details(refnr: str) -> dict:
    encoded = base64.b64encode(refnr.encode("utf-8")).decode("ascii")
    return _get(f"{BASE_URL}/jobdetails/{encoded}")


def _iso_date(value: str | None) -> str | None:
    # API liefert mal "2026-08-30", mal Zeitstempel, TTL braucht xsd:date
    if value and re.match(r"^\d{4}-\d{2}-\d{2}", value):
        return value[:10]
    return None


def to_raw_advertisement(hit: dict, detail: dict, crawl_date: str) -> RawAdvertisement | None:
    """None, wenn kein Volltext vorliegt (externes Inserat)."""
    text = (detail.get("stellenangebotsBeschreibung") or "").strip()
    refnr = hit.get("refnr") or detail.get("referenznummer")
    if not text or not refnr:
        return None
    posting_date = (
        _iso_date(hit.get("aktuelleVeroeffentlichungsdatum"))
        or _iso_date(detail.get("datumErsteVeroeffentlichung"))
        or crawl_date
    )
    return RawAdvertisement(
        id=f"ba-{re.sub(r'[^A-Za-z0-9]+', '-', refnr).strip('-').lower()}",
        source_uri=DETAIL_PAGE.format(refnr=refnr),
        crawl_date=crawl_date,
        platform=PLATFORM,
        employer=hit.get("arbeitgeber") or detail.get("firma") or "unbekannt",
        posting_date=posting_date,
        title=hit.get("titel") or hit.get("beruf") or detail.get("stellenangebotsTitel") or refnr,
        raw_text=text,
    )


def fetch(was: str, wo: str | None = None, count: int = 25) -> list[RawAdvertisement]:
    crawl_date = date.today().isoformat()
    raws: list[RawAdvertisement] = []
    page = 1
    while len(raws) < count:
        hits = search(was, wo=wo, size=min(count * 2, 100), page=page)
        if not hits:
            break
        for hit in hits:
            refnr = hit.get("refnr")
            if not refnr:
                continue
            time.sleep(REQUEST_DELAY)
            try:
                detail = details(refnr)
            except Exception as exc:  # noqa: BLE001 - einzelner Treffer soll den Lauf nicht kippen
                log.warning("jobsuche: Detail %s fehlgeschlagen (%s)", refnr, exc)
                continue
            raw = to_raw_advertisement(hit, detail, crawl_date)
            if raw is None:
                log.debug("jobsuche: %s ohne Volltext, uebersprungen", refnr)
                continue
            raws.append(raw)
            if len(raws) >= count:
                break
        page += 1
    log.info("jobsuche: %d Anzeigen mit Volltext eingesammelt", len(raws))
    return raws


def save(raws: list[RawAdvertisement], out_path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [r.model_dump(exclude_none=True) for r in raws]
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    log.info("jobsuche: %d Anzeigen -> %s", len(raws), out_path)
