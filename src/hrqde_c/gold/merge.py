"""Label-Studio-Exporte der drei Gold-Projekte zusammenfuehren (UCon3-UCas3, SA-3.03).

Die beiden unabhaengigen Annotationen entstehen in getrennten Projekten (A, B), weil
die freie Version von Label Studio jedem Bearbeiter alle Annotationen einer Aufgabe
zeigt. Die gemeinsame Fassung entsteht in einem dritten Projekt (C), das mit der
Vereinigung beider Fassungen als Vorschlaegen befuellt wird.

seed_tasks:   A, B -> Tasks fuer Projekt C (Vereinigung als predictions)
merge_exports: A, B, [C] -> ein Export mit annotations = [A, B, C] je Aufgabe,
               wie label_studio.import_gold ihn erwartet. Ohne C gilt A als
               gemeinsame Fassung (nur fuer Probelaeufe).
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def _by_raw_id(export: list[dict]) -> dict[str, dict]:
    return {t["data"]["raw_id"]: t for t in export if (t.get("data") or {}).get("raw_id")}


def _first_annotation(task: dict | None) -> dict | None:
    if not task:
        return None
    anns = [a for a in task.get("annotations") or [] if not a.get("was_cancelled")]
    return anns[0] if anns else None


def _regions(annotation: dict | None) -> list[dict]:
    if not annotation:
        return []
    return [r for r in annotation.get("result") or [] if r.get("type") == "labels"]


def seed_tasks(export_a: list[dict], export_b: list[dict]) -> list[dict]:
    """Tasks fuer das dritte Projekt: beide Fassungen als Vorschlaege, exakte Doppel einmal."""
    a, b = _by_raw_id(export_a), _by_raw_id(export_b)
    tasks = []
    for raw_id, task_a in a.items():
        seen: set[tuple] = set()
        result = []
        for ann in (_first_annotation(task_a), _first_annotation(b.get(raw_id))):
            for region in _regions(ann):
                value = region["value"]
                key = (value["start"], value["end"], tuple(value.get("labels") or []))
                if key in seen:
                    continue
                seen.add(key)
                result.append({k: region[k] for k in ("from_name", "to_name", "type", "value")})
        tasks.append({"data": task_a["data"],
                      "predictions": [{"model_version": "A+B", "result": result}]})
    log.info("merge: %d Tasks mit Vorschlaegen aus beiden Fassungen", len(tasks))
    return tasks


def merge_exports(
    export_a: list[dict], export_b: list[dict], export_c: list[dict] | None = None
) -> tuple[list[dict], list[str]]:
    """Ein Export je Aufgabe mit den Annotationen A, B und C in dieser Reihenfolge.
    -> (Export, Liste fehlender Annotationen als 'raw_id:A|B|C')"""
    a, b = _by_raw_id(export_a), _by_raw_id(export_b)
    c = _by_raw_id(export_c) if export_c is not None else None
    merged, missing = [], []
    for raw_id, task_a in a.items():
        sources = (("A", task_a), ("B", b.get(raw_id)),
                   ("C", c.get(raw_id) if c is not None else task_a))
        annotations = []
        for name, task in sources:
            ann = _first_annotation(task)
            if ann is None:
                missing.append(f"{raw_id}:{name}")
                continue
            ann = dict(ann)
            ann["completed_by"] = name
            annotations.append(ann)
        merged.append({"data": task_a["data"], "annotations": annotations})
    log.info("merge: %d Aufgaben, %d fehlende Annotationen", len(merged), len(missing))
    return merged, missing
