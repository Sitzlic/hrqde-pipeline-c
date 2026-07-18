# HR-QDE Säule C

NLP-Pipeline für Kompetenzextraktion aus Stellenanzeigen (Masterarbeit, Aufgabe C).

## Stand

Durchstich von Akquise bis TTL-Lieferung:

- NLP mit spaCy (`de_core_news_sm`)
- zwei Span-Extraktoren parallel: regelbasiert (ADJ+NOUN) und ESCOXLM-R
- ESCO-Mapping per Sentence-Encoder gegen einen Konzeptindex
- DQR-Niveau über ISCO-Hauptgruppen-Heuristik
- TTL nach HR-QDE-Lieferspec, SHACL-validiert

Akquise ist noch ein JSON-Reader, kein Crawler.

Reihenfolge der Stages in `pipeline.py`:

    acquire -> process_nlp -> extract_spans -> aggregate -> map_to_esco
    -> build_requirements -> write_ttl (+ mapping-CSV + run_metadata)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[ml]"
python -m spacy download de_core_news_sm
```

## Lauf

```bash
# Ein Minimalbeispiel
hrqde-c -i data/input/example_posting.json -o data/output -v

# Drei realistische Anzeigen (IT, Pflege, kaufmännisch)
hrqde-c -i data/input/postings_sample.json -o data/output -v
```

Pro Lauf entstehen im Output-Verzeichnis: `<id>.ttl` (Lieferung),
`<id>_mappings.csv` (alle Mapping-Kandidaten für die Stichprobenprüfung),
`run_metadata.json` (Lauf-Parameter).

### Encoder wechseln

Der ESCO-Encoder ist per Umgebungsvariable austauschbar:

```bash
HRQDE_ENCODER="isy-thl/multilingual-e5-base-course-skill-tuned" hrqde-c -i ...
```

## Verzeichnisse

- `src/hrqde_c/models.py` - Datenklassen (Pydantic)
- `src/hrqde_c/pipeline.py` - Stages und Orchestrierung
- `src/hrqde_c/extractors/` - Span-Extraktoren (adj_noun, escoxlm_r)
- `src/hrqde_c/mapping/` - ESCO-Index/Matching und DQR-Heuristik
- `src/hrqde_c/cli.py` - CLI
- `scripts/` - Probeläufe (nicht Teil der Pipeline)
- `data/input/` - Beispiel-Anzeigen (JSON)
- `data/esco/` - Konzeptindex (Mini-Bootstrap) + Embeddings-Cache
- `data/shapes/` - SHACL-Shapes des Toolkits für die Validierung
- `data/output/` - Pipeline-Output (TTL, CSV, Metadaten)

## Tests

```bash
pytest tests/
```

`test_ttl_conforms_to_hrqde_shapes` prüft die TTL-Ausgabe gegen die
HR-QDE-Shapes.
