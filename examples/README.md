# Beispiel-Durchlauf

Eine Stellenanzeige (Pflegefachkraft) einmal durch die Pipeline, zum Anschauen ohne Installation:

- `pflege-fachkraft-002_input.json` - die Eingabe-Anzeige
- `pflege-fachkraft-002.ttl` - die erzeugte Lieferung, validiert gegen `hrqde-shapes-all.ttl` (CONFORMS)
- `pflege-fachkraft-002_mappings.csv` - alle Mapping-Kandidaten inkl. der verworfenen, mit Scores

Erzeugt am 02.10.2026 mit dem vollen ESCO-Index in der Version v1.2.1 (echte ESCO-URIs, siehe Haupt-README
"ESCO-Index"). Ohne den vollen Index nutzt die Pipeline den Mini-Bestand
und liefert entsprechend Stub-URIs.

Selbst erzeugen: `hrqde-c run -i examples/pflege-fachkraft-002_input.json -o data/output -v` (Setup siehe Haupt-README).

## Zweites Beispiel: Data Engineer

Dieselbe Kette über eine IT-Anzeige (`it-data-engineer-001`), erzeugt am 02.10.2026 mit dem vollen ESCO-Index in der Version v1.2.1:

- `it-data-engineer-001_input.json` - die Eingabe-Anzeige (aus `data/input/postings_sample.json`)
- `it-data-engineer-001.ttl` - die Lieferung, validiert gegen `hrqde-shapes-all.ttl` (CONFORMS)
- `it-data-engineer-001_mappings.csv` - alle Mapping-Kandidaten inkl. der verworfenen, mit Scores

Selbst erzeugen: `hrqde-c run -i examples/it-data-engineer-001_input.json -o data/output -v`
