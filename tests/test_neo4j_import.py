"""Integrationstest, laeuft nur gegen eine erreichbare Neo4j-Instanz.

Aktivieren: docker compose up -d, dann HRQDE_NEO4J=1 pytest tests/test_neo4j_import.py
"""

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("HRQDE_NEO4J"),
    reason="braucht laufende Neo4j-Instanz (HRQDE_NEO4J=1 setzen)",
)


def test_import_delivery_roundtrip(tmp_path: Path):
    from hrqde_c import pipeline
    from hrqde_c.graph import neo4j_import

    outputs = pipeline.run(Path("data/input/example_posting.json"), tmp_path / "out")
    files, triples = neo4j_import.load_directory(
        tmp_path / "out", "bolt://localhost:7687", "neo4j", "hrqde-dev"
    )
    assert files == len(outputs)
    assert triples > 0

    from neo4j import GraphDatabase
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "hrqde-dev"))
    with driver.session() as session:
        # n10s praefixt die Vokabular-Labels mit dem registrierten hrqde-Prefix
        n = session.run(
            "MATCH (p:hrqde__JobPosting)-[:hrqde__hasRequirement]->"
            "(q:hrqde__QualificationRequirement) RETURN count(q) AS n"
        ).single()["n"]
    driver.close()
    assert n > 0
