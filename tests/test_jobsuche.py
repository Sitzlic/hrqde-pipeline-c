from hrqde_c.acquisition import jobsuche

SEARCH_HIT = {
    "referenznummer": "17606-43991424-223-S",
    "stellenangebotsTitel": "Duales Studium Informatik (m/w/d)",
    "firma": "Biotronik SE & Co. KG",
    "datumErsteVeroeffentlichung": "2026-08-30",
}

DETAIL = {
    "referenznummer": "17606-43991424-223-S",
    "stellenangebotsBeschreibung": "Gestalte die Zukunft der Medizintechnik! ...",
    "datumErsteVeroeffentlichung": "2026-08-15T00:00:00+02:00",
}


def test_mapping_builds_raw_advertisement():
    raw = jobsuche.to_raw_advertisement(SEARCH_HIT, DETAIL, crawl_date="2026-09-06")
    assert raw is not None
    assert raw.id == "ba-17606-43991424-223-s"
    assert raw.platform == "ba_jobsuche"
    assert raw.employer == "Biotronik SE & Co. KG"
    assert raw.posting_date == "2026-08-30"
    assert raw.source_uri.endswith("17606-43991424-223-S")
    assert raw.raw_text.startswith("Gestalte")


def test_posting_date_falls_back_to_detail_timestamp():
    hit = {**SEARCH_HIT, "datumErsteVeroeffentlichung": None}
    raw = jobsuche.to_raw_advertisement(hit, DETAIL, crawl_date="2026-09-06")
    assert raw.posting_date == "2026-08-15"


def test_external_posting_without_text_is_skipped():
    raw = jobsuche.to_raw_advertisement(SEARCH_HIT, {"externeUrl": "https://..."},
                                        crawl_date="2026-09-06")
    assert raw is None


def test_fetch_logs_skips_and_balance(monkeypatch, caplog):
    import logging

    hits = [SEARCH_HIT, {**SEARCH_HIT, "referenznummer": "extern-1"},
            {**SEARCH_HIT, "referenznummer": "fehler-1"}]
    pages = iter([hits, []])
    monkeypatch.setattr(jobsuche, "search", lambda *a, **k: next(pages))
    monkeypatch.setattr(jobsuche, "REQUEST_DELAY", 0)

    def fake_details(refnr):
        if refnr == "fehler-1":
            raise OSError("timeout")
        if refnr == "extern-1":
            return {"externeUrl": "https://..."}
        return DETAIL

    monkeypatch.setattr(jobsuche, "details", fake_details)
    with caplog.at_level(logging.INFO, logger="hrqde_c.acquisition.jobsuche"):
        raws, stats = jobsuche.fetch_with_stats("Informatik", count=5)
    assert len(raws) == 1
    assert stats["hits"] == 3 and stats["fetched"] == 1
    assert stats["skipped"] == 1 and stats["errors"] == 1
    assert "extern-1 uebersprungen, kein Volltext" in caplog.text
    assert "Bilanz 3 Treffer, 1 abgerufen, 1 uebersprungen, 1 Fehler" in caplog.text


def test_save_fetch_log_next_to_output(tmp_path):
    import json

    out = tmp_path / "fetched.json"
    path = jobsuche.save_fetch_log({"was": "x", "hits": 2}, out)
    assert path == tmp_path / "fetched.fetch.json"
    assert json.loads(path.read_text(encoding="utf-8"))["output"] == str(out)
