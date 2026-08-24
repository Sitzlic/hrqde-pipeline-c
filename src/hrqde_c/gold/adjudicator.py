"""Cohen's Kappa ueber doppelt annotierte Anzeigen (SA-C.3.03)."""

from __future__ import annotations

from collections import Counter


def cohens_kappa(a: list[str], b: list[str]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("Label-Folgen muessen gleich lang und nicht leer sein")
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    count_a, count_b = Counter(a), Counter(b)
    expected = sum(
        (count_a[cat] / n) * (count_b[cat] / n) for cat in set(a) | set(b)
    )
    if expected == 1.0:
        # beide Annotatoren nutzen nur eine Kategorie und stimmen ueberein
        return 1.0
    return (observed - expected) / (1 - expected)
