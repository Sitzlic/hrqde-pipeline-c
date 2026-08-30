# HR-QDE Säule C

NLP-Pipeline für Kompetenzextraktion aus Stellenanzeigen (Masterarbeit, Aufgabe C).

## Stand

Durchstich von Akquise bis TTL-Lieferung:

- NLP mit spaCy (`de_core_news_sm`)
- zwei Span-Extraktoren parallel: regelbasiert (ADJ+NOUN) und ESCOXLM-R
- ESCO-Mapping per Sentence-Encoder gegen einen Konzeptindex
- DQR-Niveau über ISCO-Hauptgruppen-Heuristik
- TTL nach HR-QDE-Lieferspec, SHACL-validiert

Akquise: JSON-Reader plus Anbindung der BA-Jobsuche-API (`fetch`-Kommando,
funktioniert nur mit deutscher IP, die API blockt Auslands-Anfragen).

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
hrqde-c run -i data/input/example_posting.json -o data/output -v

# Drei realistische Anzeigen (IT, Pflege, kaufmännisch)
hrqde-c run -i data/input/postings_sample.json -o data/output -v

# Echte Anzeigen von der BA-Jobsuche holen (deutsche IP noetig)
hrqde-c fetch --was "Pflegefachkraft" --wo Berlin -n 25 -o data/input/fetched.json
```

Pro Lauf entstehen im Output-Verzeichnis: `<id>.ttl` (Lieferung),
`<id>_mappings.csv` (alle Mapping-Kandidaten für die Stichprobenprüfung),
`run_metadata.json` (Lauf-Parameter).

### ESCO-Index

Mitgeliefert ist nur ein Mini-Bestand (`data/esco/skills_mini.json`, 40 Konzepte).
Den vollen Index (alle ~13.500 ESCO-Skills, Labels DE+EN) einmalig bauen:

```bash
python scripts/fetch_esco_skills.py
```

Liegt `data/esco/skills_full.json` danach vor, nutzt die Pipeline ihn automatisch.
Beim ersten Lauf wird der Embedding-Cache gebaut, das dauert ein paar Minuten.
Übersteuern per `HRQDE_ESCO_CONCEPTS=<pfad>`.

### Encoder wechseln

Der ESCO-Encoder ist per Umgebungsvariable austauschbar:

```bash
HRQDE_ENCODER="isy-thl/multilingual-e5-base-course-skill-tuned" hrqde-c run -i ...
```

## Gold Standard

Werkzeugkette fuer den annotierten Referenzbestand:

```bash
# 1. Stichprobe ziehen: Kalibrier-/Test-Split + Label-Studio-Tasks
#    mit Vor-Annotationen aus beiden Extraktoren
hrqde-c gold-sample -i data/input/fetched.json -n 60 --seed 42

# 2. data/gold/labelstudio_config.xml als Labeling-Interface in Label Studio
#    einrichten, labelstudio_tasks.json importieren, annotieren, Export als JSON

# 3. Export uebernehmen (BIO-Schema, bei Doppelannotation Cohen's Kappa)
hrqde-c gold-import -e ls_export.json
```

Der Split steht im `sample_manifest.json` und wird beim Import wieder
angelegt: Kalibrierteil fuer das Threshold-Tuning, Testteil fuer den
Extraktor-Vergleich.

Auswertung gegen den fertigen Gold Standard:

```bash
# Span-F1 pro Extraktionsvariante gegen den Testteil, Bericht als CSV
hrqde-c evaluate -i data/input/fetched.json

# Schwellenwert-Tabelle auf dem Kalibrierteil
# (braucht GoldSpans mit manuell zugeordneter esco_uri)
hrqde-c calibrate
```

Die Variante ADJ+NOUN nach Uebersetzung ins Englische ist konzipiert,
aber im Prototyp nicht realisiert (deklarierte Abweichung), der
Evaluator laeuft deshalb mit zwei Varianten.

## Verzeichnisse

- `src/hrqde_c/models.py` - Datenklassen (Pydantic)
- `src/hrqde_c/pipeline.py` - Stages und Orchestrierung
- `src/hrqde_c/extractors/` - Span-Extraktoren (adj_noun, escoxlm_r)
- `src/hrqde_c/mapping/` - ESCO-Index/Matching und DQR-Heuristik
- `src/hrqde_c/gold/` - Gold-Standard-Kette (Sampling, Label Studio, BIO, Kappa)
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
