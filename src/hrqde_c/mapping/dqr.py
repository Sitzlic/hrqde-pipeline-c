from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# DQR-Default pro ISCO-Hauptgruppe, Tabelle aus der Konzeption (RC-C.2.5)
ISCO_MAJOR_GROUP_DEFAULT: dict[int, str] = {
    1: "DQR7",  # Managers
    2: "DQR6",  # Professionals
    3: "DQR5",  # Technicians & Associates
    4: "DQR4",  # Clerical
    5: "DQR4",  # Service & Sales
    6: "DQR3",  # Skilled Agricultural
    7: "DQR3",  # Craft & Trades
    8: "DQR3",  # Plant & Machine Operators
    9: "DQR2",  # Elementary
}

# DQR4 = abgeschlossene Ausbildung, konservative Mitte
FALLBACK_LEVEL = "DQR4"


def default_for_isco_group(isco_group: int | None) -> str:
    """Hauptgruppen-Default als Rueckfall, solange der Occupation-Lookup fehlt."""
    if isco_group is None:
        return FALLBACK_LEVEL
    level = ISCO_MAJOR_GROUP_DEFAULT.get(isco_group)
    if level is None:
        log.warning("dqr: unbekannte ISCO-Hauptgruppe %r, nutze Fallback", isco_group)
        return FALLBACK_LEVEL
    return level
