# HR-QDE Säule C

NLP-Pipeline für Kompetenzextraktion aus Stellenanzeigen (Masterarbeit, Aufgabe C).

## Stand

Durchstich von Akquise bis TTL-Lieferung:

- NLP mit spaCy (`de_core_news_sm`)
- zwei Span-Extraktoren parallel: regelbasiert (ADJ+NOUN) und ESCOXLM-R
- ESCO-Mapping per Sentence-Encoder gegen einen Konzeptindex
- DQR-Niveau über ISCO-Hauptgruppen-Heuristik
- TTL nach HR-QDE-Lieferspec, SHACL-validiert

Akquise: JSON-Reader plus Anbindung der BA-Jobsuche-API (`fetch`-Kommando).

Reihenfolge der Stages in `pipeline.py`:

    acquire -> process_nlp -> extract_spans -> aggregate -> map_to_esco
    -> build_requirements -> write_ttl (+ mapping-CSV, drafts.json, span_candidates.csv, run_metadata)

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
hrqde-c run -i data/input/example_posting.json -o data/output -v
# Schwellenwert des Mapping-Filters setzen (Standard 0.5, Bereich 0..1):
hrqde-c run -i data/input/example_posting.json -o data/output --threshold 0.65

# Drei realistische Anzeigen (IT, Pflege, kaufmännisch)
hrqde-c run -i data/input/postings_sample.json -o data/output -v

# Echte Anzeigen von der BA-Jobsuche holen; daneben entsteht
# data/input/fetched.fetch.json mit Abrufparametern und Bilanz
hrqde-c fetch --was "Pflegefachkraft" --wo Berlin -n 25 -o data/input/fetched.json

# TTL-Lieferung eines Verzeichnisses gegen die HR-QDE-Shapes pruefen (SHACL)
hrqde-c validate -i data/output
```

Pro Lauf entstehen im Output-Verzeichnis: `<id>.ttl` (Lieferung),
`<id>_mappings.csv` (alle Mapping-Kandidaten für die Stichprobenprüfung),
`drafts.json` (Demand-Bestand des Laufs, eine JobPostingDraft je Anzeige),
`span_candidates.csv` (alle Span-Kandidaten beider Extraktoren mit Bezug auf
die Quellanzeige), `run_metadata.json` (Lauf-Parameter).

### ESCO-Index

Mitgeliefert ist nur ein Mini-Bestand (`data/esco/skills_mini.json`, 40 Konzepte).
Den vollen Index (13.939 ESCO-Skills der Version v1.2.1, Labels DE+EN) und die
Berufsprofile fuer die DQR-Heuristik (3.039 Occupations) einmalig holen:

```bash
python scripts/fetch_esco_skills.py
python scripts/fetch_esco_occupations.py
```

Beide Skripte fragen die ESCO-API mit `selectedVersion=v1.2.1` ab. Ohne diesen
Parameter liefert die API nicht die aktuelle Version, sondern einen aelteren
Bestand (13.485 Skills, 2.942 Occupations).

Liegt `data/esco/skills_full.json` danach vor, nutzt die Pipeline ihn automatisch.
Beim ersten Lauf wird der Embedding-Cache gebaut, das dauert ein paar Minuten.
Übersteuern per `HRQDE_ESCO_CONCEPTS=<pfad>`.

### Encoder wechseln

Der ESCO-Encoder ist per Umgebungsvariable austauschbar:

```bash
HRQDE_ENCODER="isy-thl/multilingual-e5-base-course-skill-tuned" hrqde-c run -i ...
```

## Neo4j-Ablage (optional)

Die TTL-Lieferung laesst sich ueber neosemantics in einen Graphen laden:

```bash
docker compose up -d                 # Neo4j 5 mit n10s-Plugin (und Label Studio)
pip install '.[graph]'               # neo4j-Treiber
hrqde-c run -i data/input/... -o data/output
hrqde-c load-neo4j -i data/output    # TTL -> Graph
```

Browser unter http://localhost:7474 (neo4j / hrqde-dev). Der Import laeuft
ueber `n10s.rdf.import.inline`, die Vokabular-Labels tragen den
registrierten Prefix `hrqde__` (z. B. `hrqde__JobPosting`).

## Gold Standard

Werkzeugkette fuer den annotierten Referenzbestand:

```bash
# 1. Stichprobe ziehen: Kalibrier-/Test-Split + Label-Studio-Tasks
#    mit Vor-Annotationen aus beiden Extraktoren
hrqde-c gold-sample -i data/input/fetched.json -n 60 --seed 42

# 2. data/gold/labelstudio_config.xml als Labeling-Interface in Label Studio
#    einrichten, labelstudio_tasks.json importieren, annotieren, Export als JSON

# 3. Export uebernehmen (BIO-Schema, bei Doppelannotation Cohen's Kappa ueber
#    die ersten beiden Annotationen; eine dritte Annotation gilt als die
#    zusammengefuehrte Fassung und liefert die Gold-Spans)
hrqde-c gold-merge --seed export_a.json export_b.json -o tasks_c.json   # Tasks fuer die gemeinsame Fassung
hrqde-c gold-merge export_a.json export_b.json export_c.json -o ls_export.json
hrqde-c gold-import -e ls_export.json
```

Der Split steht im `sample_manifest.json` und wird beim Import wieder
angelegt: Kalibrierteil fuer das Threshold-Tuning, Testteil fuer den
Extraktor-Vergleich.

Auswertung gegen den fertigen Gold Standard:

```bash
# Span-F1 pro Extraktionsvariante gegen den Testteil, Bericht als CSV
hrqde-c evaluate -i data/input/fetched.json

# Referenz-URIs fuer den Kalibrierteil: Top-3-Vorschlaege des Mappers je
# Gold-Span als CSV, in der Spalte decision Rang (1-3), eigene URI oder "-"
# (kein passendes Konzept) eintragen, dann zurueckspielen
hrqde-c gold-esco-candidates
hrqde-c gold-esco-import -d data/gold/esco_candidates.csv

# Schwellenwert-Tabelle auf dem Kalibrierteil
# (braucht GoldSpans mit esco_uri, "-" zaehlt als Fehlgriff bei Akzeptanz)
hrqde-c calibrate
```

Die dritte Variante ADJ+NOUN nach Uebersetzung (`extractors/adj_noun_en.py`,
Helsinki-NLP/opus-mt-de-en, braucht `python -m spacy download en_core_web_sm`)
liefert Spans auf dem uebersetzten Text. Ihre Offsets sind deshalb nicht
gegen den deutschen Gold Standard vergleichbar (dafuer fehlt ein
Wort-Alignment), der Span-F1-Vergleich laeuft mit zwei Varianten.

## Verzeichnisse

- `src/hrqde_c/models.py` - Datenklassen (Pydantic)
- `src/hrqde_c/pipeline.py` - Stages und Orchestrierung
- `src/hrqde_c/extractors/` - Span-Extraktoren (adj_noun, escoxlm_r)
- `src/hrqde_c/mapping/` - ESCO-Index/Matching und DQR-Heuristik
- `src/hrqde_c/gold/` - Gold-Standard-Kette (Sampling, Label Studio, BIO, Kappa)
- `src/hrqde_c/graph/` - Neo4j-Import der TTL-Lieferung (neosemantics)
- `src/hrqde_c/acquisition/` - BA-Jobsuche-Anbindung
- `src/hrqde_c/cli.py` - CLI
- `scripts/` - Probeläufe (nicht Teil der Pipeline)
- `data/input/` - Beispiel-Anzeigen (JSON)
- `data/esco/` - Konzeptindex (Mini-Bootstrap, voller Index per Script) + Embeddings-Cache
- `data/shapes/` - SHACL-Shapes des Toolkits für die Validierung
- `data/output/` - Pipeline-Output (TTL, CSV, Metadaten)

## Beispiel

Ein kompletter Durchlauf (Anzeige -> TTL -> Mapping-CSV) liegt in [`examples/`](examples/).

## Tests

```bash
pytest tests/
```

`test_ttl_conforms_to_hrqde_shapes` prüft die TTL-Ausgabe gegen die
HR-QDE-Shapes.
