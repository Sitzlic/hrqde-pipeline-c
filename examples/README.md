# Beispiel-Durchlauf

Eine Stellenanzeige (Pflegefachkraft) einmal durch die Pipeline, zum Anschauen ohne Installation:

- `pflege-fachkraft-002_input.json` - die Eingabe-Anzeige
- `pflege-fachkraft-002.ttl` - die erzeugte Lieferung, validiert gegen `hrqde-shapes-all.ttl` (CONFORMS)
- `pflege-fachkraft-002_mappings.csv` - alle Mapping-Kandidaten inkl. der verworfenen, mit Scores

Erzeugt mit dem vollen ESCO-Index (echte ESCO-URIs, siehe Haupt-README
"ESCO-Index"). Ohne den vollen Index nutzt die Pipeline den Mini-Bestand
und liefert entsprechend Stub-URIs.

Selbst erzeugen: `hrqde-c run -i examples/pflege-fachkraft-002_input.json -o data/output -v` (Setup siehe Haupt-README).
