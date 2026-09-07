import json

from hrqde_c.mapping import dqr
from hrqde_c.models import EscoOccupation


def _occ(uri, group, skills):
    return EscoOccupation(uri=uri, pref_label=uri, code=f"{group}223.1",
                          isco_major_group=group, essential_skills=skills)


def test_lookup_majority_wins():
    occs = (
        _occ("occ:pflege", 3, ["s:1", "s:2", "s:3"]),
        _occ("occ:it", 2, ["s:9"]),
    )
    assert dqr.lookup(["s:1", "s:2", "s:9"], occs).uri == "occ:pflege"


def test_lookup_requires_majority_and_unique_top():
    occs = (
        _occ("occ:a", 2, ["s:1"]),
        _occ("occ:b", 3, ["s:2"]),
    )
    # Gleichstand (je 1 Stimme) -> kein eindeutiger Treffer
    assert dqr.lookup(["s:1", "s:2"], occs) is None
    # 1 von 3 ist keine Mehrheit
    assert dqr.lookup(["s:1", "s:8", "s:9"], (occs[0],)) is None


def test_determine_three_stages(tmp_path, monkeypatch):
    occ_file = tmp_path / "occs.json"
    occ_file.write_text(json.dumps([{
        "uri": "http://data.europa.eu/esco/occupation/x", "pref_label": "Testberuf",
        "code": "2512.1", "isco_major_group": 2, "essential_skills": ["s:1", "s:2"],
    }]))
    monkeypatch.setenv(dqr.OCCUPATIONS_ENV, str(occ_file))
    dqr._load_occupations.cache_clear()

    # Stufe 1: Mehrheit der Konzepte in den Essential Skills
    r = dqr.determine(["s:1", "s:2"], isco_group=9)
    assert (r.level, r.stage) == ("DQR6", "occupation-lookup")
    assert r.occupation_uri == "http://data.europa.eu/esco/occupation/x"

    # Stufe 2: kein Lookup-Treffer, ISCO-Gruppe der Anzeige greift
    r = dqr.determine(["s:8", "s:9"], isco_group=9)
    assert (r.level, r.occupation_uri, r.stage) == ("DQR2", None, "isco-default")

    # Stufe 3: weder Treffer noch ISCO-Gruppe
    r = dqr.determine([], isco_group=None)
    assert (r.level, r.occupation_uri, r.stage) == ("DQR4", None, "fallback")

    dqr._load_occupations.cache_clear()
