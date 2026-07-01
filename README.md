# HR-QDE Säule C

NLP-Pipeline für Kompetenzextraktion aus Stellenanzeigen (Masterarbeit, Aufgabe C).

## Stand

Durchstich von Akquise bis TTL-Lieferung. NLP läuft mit spaCy (`de_core_news_sm`),
Span-Extraktion und ESCO-Mapping sind noch Platzhalter.

Reihenfolge der Stages in `pipeline.py`:

    acquire -> process_nlp -> extract_spans -> aggregate -> map_to_esco
    -> build_requirements -> write_ttl

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m spacy download de_core_news_sm
```

## Lauf

```bash
hrqde-c -i data/input/example_posting.json -o data/output -v
```

## Verzeichnisse

- `src/hrqde_c/models.py` - Datenklassen (Pydantic)
- `src/hrqde_c/pipeline.py` - Stages und Orchestrierung
- `src/hrqde_c/cli.py` - CLI
- `data/input/` - Beispiel-Anzeigen (JSON)
- `data/output/` - Pipeline-Output (TTL)
