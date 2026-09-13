from hrqde_c.mapping import dqr
from hrqde_c.models import EscoOccupation


def _occ(uri, group, label):
    return EscoOccupation(uri=uri, pref_label=label, code=f"{group}223.1",
                          isco_major_group=group, essential_skills=[])


def _index(labels_groups):
    occs = tuple(_occ(f"occ:{i}", g, lbl) for i, (lbl, g) in enumerate(labels_groups))
    emb = dqr.esco.encode([o.pref_label for o in occs], mode="passage")
    return occs, emb


def test_normalize_title_strips_gender_tags():
    assert dqr._normalize_title("Softwareentwickler (m/w/d)") == "Softwareentwickler"
    assert dqr._normalize_title("Buchhalter/in") == "Buchhalter"


def test_lookup_matches_clear_title():
    occs, emb = _index([("Softwareentwickler", 2), ("Krankenpfleger", 5)])
    hit = dqr.lookup("Softwareentwickler (m/w/d)", occs, emb)
    assert hit is not None and hit.isco_major_group == 2


def test_lookup_returns_none_below_threshold():
    # ein voellig fremder Titel darf kein Profil erzwingen
    occs, emb = _index([("Softwareentwickler", 2), ("Krankenpfleger", 5)])
    assert dqr.lookup("Zoologischer Garteninspektor", occs, emb) is None


def test_lookup_empty_index():
    assert dqr.lookup("Softwareentwickler", (), None) is None


def test_determine_three_stages(monkeypatch):
    occs, emb = _index([("Softwareentwickler", 2)])
    monkeypatch.setattr(dqr, "_occupation_index", lambda *a: (occs, emb))

    r = dqr.determine("Softwareentwickler (m/w/d)", isco_group=9)
    assert (r.level, r.stage) == ("DQR6", "occupation-lookup")
    assert r.occupation_uri == "occ:0"

    # kein Titel-Treffer -> ISCO-Gruppe der Anzeige
    r = dqr.determine("Zoologischer Garteninspektor", isco_group=9)
    assert (r.level, r.occupation_uri, r.stage) == ("DQR2", None, "isco-default")

    # weder Treffer noch ISCO-Gruppe
    r = dqr.determine("Zoologischer Garteninspektor", isco_group=None)
    assert (r.level, r.occupation_uri, r.stage) == ("DQR4", None, "fallback")
