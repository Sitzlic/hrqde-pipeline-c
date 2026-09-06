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
