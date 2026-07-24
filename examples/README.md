# Beispiel-Durchlauf

Eine Stellenanzeige (Pflegefachkraft) einmal durch die Pipeline, zum Anschauen ohne Installation:

- `pflege-fachkraft-002_input.json` - die Eingabe-Anzeige
- `pflege-fachkraft-002.ttl` - die erzeugte Lieferung, validiert gegen `hrqde-shapes-all.ttl` (CONFORMS)
- `pflege-fachkraft-002_mappings.csv` - alle Mapping-Kandidaten inkl. der verworfenen, mit Scores

Selbst erzeugen: `hrqde-c -i data/input/postings_sample.json -o data/output -v` (Setup siehe Haupt-README).
