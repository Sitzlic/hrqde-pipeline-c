"""Ablage der TTL-Lieferung in Neo4j ueber neosemantics (RC-C.2.3).

Der Import laeuft ueber n10s.rdf.import.inline, die TTL-Dateien werden
also nicht in den Container gemountet, sondern als Payload uebergeben.
Einmalige Initialisierung: Uniqueness-Constraint auf Resource.uri und
n10s-Graphconfig (Labels aus rdf:type, URIs behalten).
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

INIT_CONSTRAINT = (
    "CREATE CONSTRAINT n10s_unique_uri IF NOT EXISTS "
    "FOR (r:Resource) REQUIRE r.uri IS UNIQUE"
)


def _connect(uri: str, user: str, password: str):
    # lazy import, der Treiber ist ein optionales Extra ([graph])
    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise RuntimeError(
            "neo4j-Treiber fehlt. Installieren mit: pip install '.[graph]'"
        ) from exc
    # Server-Notifications des Treibers (z. B. Hinweise zu Cypher-Statements von
    # n10s) nicht auf der Konsole ausgeben, Fehler bleiben sichtbar
    logging.getLogger("neo4j").setLevel(logging.ERROR)
    return GraphDatabase.driver(uri, auth=(user, password))


def init_graph(session) -> None:
    session.run(INIT_CONSTRAINT)
    ready = session.run(
        "MATCH (c:`_GraphConfig`) RETURN count(c) AS n"
    ).single()["n"]
    if ready == 0:
        session.run(
            "CALL n10s.graphconfig.init({handleVocabUris: 'SHORTEN', "
            "keepLangTag: false, handleMultival: 'OVERWRITE'})"
        )
        # festen Prefix registrieren, sonst vergibt n10s ns0/ns1 in Ladereihenfolge
        session.run(
            "CALL n10s.nsprefixes.add('hrqde', 'http://hr-qde.org/ontology/')"
        )
        log.info("neo4j: n10s-Graphconfig initialisiert (Prefix hrqde)")


def import_ttl(session, ttl_path: Path) -> int:
    payload = ttl_path.read_text(encoding="utf-8")
    record = session.run(
        "CALL n10s.rdf.import.inline($payload, 'Turtle')",
        payload=payload,
    ).single()
    status = record["terminationStatus"]
    if status != "OK":
        raise RuntimeError(f"{ttl_path.name}: Import fehlgeschlagen ({record['extraInfo']})")
    count = record["triplesLoaded"]
    log.info("neo4j: %s -> %d Tripel", ttl_path.name, count)
    return count


def load_directory(
    input_dir: Path, uri: str, user: str, password: str
) -> tuple[int, int]:
    """Importiert alle TTL-Dateien eines Verzeichnisses. -> (Dateien, Tripel)"""
    ttl_files = sorted(input_dir.glob("*.ttl"))
    if not ttl_files:
        raise RuntimeError(f"keine TTL-Dateien in {input_dir}")
    driver = _connect(uri, user, password)
    triples = 0
    with driver.session() as session:
        init_graph(session)
        for ttl in ttl_files:
            triples += import_ttl(session, ttl)
    driver.close()
    log.info("neo4j: %d Dateien, %d Tripel importiert", len(ttl_files), triples)
    return len(ttl_files), triples


def graph_size(uri: str, user: str, password: str) -> tuple[int, int]:
    """Bestand im Graphen nach dem Laden. -> (Knoten, Beziehungen)"""
    driver = _connect(uri, user, password)
    with driver.session() as session:
        nodes = session.run("MATCH (n) RETURN count(n) AS n").single()["n"]
        rels = session.run("MATCH ()-[r]->() RETURN count(r) AS n").single()["n"]
    driver.close()
    return nodes, rels
