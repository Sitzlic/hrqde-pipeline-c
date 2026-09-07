import os

# Tests laufen gegen den Mini-Bestand, der volle Index braucht Minuten zum Bauen
os.environ.setdefault("HRQDE_ESCO_CONCEPTS", "data/esco/skills_mini.json")
os.environ.setdefault("HRQDE_ESCO_OCCUPATIONS", "data/esco/occupations_test_missing.json")
