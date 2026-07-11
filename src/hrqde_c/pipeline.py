from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import spacy
from spacy.language import Language

from hrqde_c import __version__
from hrqde_c.extractors import adj_noun, escoxlm_r
from hrqde_c.mapping import esco
from hrqde_c.models import (
    JobPostingDraft,
    MappingDecision,
    ProcessedAdvertisement,
    QualificationRequirement,
    RawAdvertisement,
    RequirementKind,
    SpanCandidate,
    SpanMapping,
    Token,
)

log = logging.getLogger(__name__)

SPACY_MODEL = "de_core_news_sm"
MAPPING_THRESHOLD = 0.5
# Begruendeter Default gemaess Lieferspec §5; Kalibrierung pro
# ISCO-Hauptgruppe folgt mit der DQR-Heuristik (RC-C.2.5).
DEFAULT_DQR_LEVEL = "DQR4"

# Signalwoerter fuer nice_to_have (Lieferspec §4)
NICE_TO_HAVE_MARKERS = re.compile(
    r"erwünscht|wünschenswert|von vorteil|idealerweise|ein plus|"
    r"bevorzugt|optional|nice to have|preferred|would be a plus",
    re.IGNORECASE,
)


@lru_cache(maxsize=1)
def _load_nlp() -> Language:
    try:
        return spacy.load(SPACY_MODEL)
    except OSError as exc:
        raise RuntimeError(
            f"spaCy-Modell '{SPACY_MODEL}' nicht installiert. "
            f"Installieren mit: python -m spacy download {SPACY_MODEL}"
        ) from exc


def acquire(input_path: Path) -> list[RawAdvertisement]:
    log.info("acquire: lese %s", input_path)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    items = payload if isinstance(payload, list) else [payload]
    raws = [RawAdvertisement(**item) for item in items]
    log.info("acquire: %d Stellenanzeigen geladen", len(raws))
    return raws


def process_nlp(raw: RawAdvertisement) -> ProcessedAdvertisement:
    nlp = _load_nlp()
    doc = nlp(raw.raw_text)
    tokens = [
        Token(text=tok.text, pos=tok.pos_, lemma=tok.lemma_)
        for tok in doc
        if not tok.is_space
    ]
    log.info("nlp: %s -> %d Tokens", raw.id, len(tokens))
    return ProcessedAdvertisement(raw_id=raw.id, language="de", tokens=tokens)


def classify_requirement_kind(raw_text: str, spans: list[SpanCandidate]) -> None:
    """Saetze mit Markern wie "von Vorteil": alle Spans darin werden nice_to_have."""
    nlp = _load_nlp()
    doc = nlp(raw_text)
    nice_ranges = [
        (sent.start_char, sent.end_char)
        for sent in doc.sents
        if NICE_TO_HAVE_MARKERS.search(sent.text)
    ]
    if not nice_ranges:
        return
    for span in spans:
        for start, end in nice_ranges:
            if start <= span.char_start < end:
                span.requirement_kind = RequirementKind.NICE_TO_HAVE
                break


def extract_spans(
    raw: RawAdvertisement, processed: ProcessedAdvertisement
) -> list[SpanCandidate]:
    # Beide Extraktionspfade laufen parallel und werden nicht zusammengelegt
    # (SA-C.1.02) - das extractor-Feld haelt sie fuer UC-C.3 unterscheidbar.
    spans = adj_noun.extract(raw, _load_nlp()) + escoxlm_r.extract(raw)
    classify_requirement_kind(raw.raw_text, spans)
    by_extractor = {"adj_noun": 0, "escoxlm_r": 0}
    for s in spans:
        by_extractor[s.extractor] += 1
    log.info(
        "extraction: %s -> %d Span-Kandidaten (adj_noun=%d, escoxlm_r=%d)",
        raw.id, len(spans), by_extractor["adj_noun"], by_extractor["escoxlm_r"],
    )
    return spans


def aggregate(raw: RawAdvertisement, spans: list[SpanCandidate]) -> JobPostingDraft:
    draft = JobPostingDraft(
        id=f"draft-{raw.id}",
        raw_advertisement=raw,
        span_candidates=spans,
    )
    log.info("aggregate: %s -> JobPostingDraft mit %d Spans", raw.id, len(spans))
    return draft


def map_to_esco(spans: list[SpanCandidate]) -> list[SpanMapping]:
    index = esco.get_index()
    mappings = []
    for span in spans:
        best = index.match(span.text, top_k=1)[0]
        mappings.append(
            SpanMapping(
                id=f"mapping-{uuid.uuid4().hex[:8]}",
                span_id=span.id,
                concept_uri=best.concept.uri,
                score=best.score,
                decision=MappingDecision.ACCEPTED
                if best.score >= MAPPING_THRESHOLD
                else MappingDecision.REJECTED,
            )
        )
    log.info("mapping: %d Spans -> %d SpanMappings", len(spans), len(mappings))
    return mappings


def build_requirements(
    spans: list[SpanCandidate], mappings: list[SpanMapping]
) -> list[QualificationRequirement]:
    by_span = {m.span_id: m for m in mappings if m.decision == MappingDecision.ACCEPTED}

    # Dubletten pro Konzept-URI: hoechster Score gewinnt, must schlaegt nice_to_have
    best_per_concept: dict[str, tuple[SpanCandidate, SpanMapping]] = {}
    for span in spans:
        mapping = by_span.get(span.id)
        if mapping is None:
            continue
        current = best_per_concept.get(mapping.concept_uri)
        if current is None or mapping.score > current[1].score:
            best_per_concept[mapping.concept_uri] = (span, mapping)
        elif current[0].requirement_kind == RequirementKind.NICE_TO_HAVE and \
                span.requirement_kind == RequirementKind.MUST:
            best_per_concept[mapping.concept_uri] = (span, mapping)

    requirements = []
    for concept_uri, (span, mapping) in best_per_concept.items():
        requirements.append(
            QualificationRequirement(
                id=f"req-{uuid.uuid4().hex[:8]}",
                refers_to_competence=concept_uri,
                required_level=DEFAULT_DQR_LEVEL,  # TODO DQR-Heuristik (RC-C.2.5)
                requirement_kind=span.requirement_kind,
                provenance_confidence=mapping.score,
            )
        )
    log.info(
        "requirements: %d Spans -> %d Requirements (konsolidiert)",
        len(spans), len(requirements),
    )
    return requirements


# --- TTL-Lieferung gemaess hrqde-shape-demand.ttl (Toolkit v0.1.1) ---

TTL_PREFIX = """@prefix hrqde:      <http://hr-qde.org/ontology/> .
@prefix sitzler:    <http://hr-qde.org/data/sitzler/> .
@prefix employer:   <http://hr-qde.org/data/sitzler/employer/> .
@prefix esco-skill: <http://data.europa.eu/esco/skill/> .
@prefix rdfs:       <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd:        <http://www.w3.org/2001/XMLSchema#> .

"""


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "unbenannt"


def _posting_uuid(raw: RawAdvertisement) -> str:
    # uuid5 ueber die Quell-URL, gleiche Anzeige ergibt gleiche ID (Lieferspec D1)
    return uuid.uuid5(uuid.NAMESPACE_URL, raw.source_uri).hex[:8]


def write_ttl(
    raw: RawAdvertisement,
    requirements: list[QualificationRequirement],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{raw.id}.ttl"

    posting_id = _posting_uuid(raw)
    employer_slug = _slugify(raw.employer)
    req_uris = [f"sitzler:req_{posting_id}_{i}" for i in range(1, len(requirements) + 1)]

    lines = [TTL_PREFIX]

    lines.append(f"employer:{employer_slug}")
    lines.append("  a           hrqde:Employer ;")
    lines.append(f'  rdfs:label  "{raw.employer}"@de .')
    lines.append("")

    lines.append(f"sitzler:posting_{posting_id}")
    lines.append("  a                    hrqde:JobPosting ;")
    lines.append(f'  rdfs:label           "{raw.title}"@de ;')
    lines.append(f'  hrqde:postingDate    "{raw.posting_date}"^^xsd:date ;')
    lines.append(f"  hrqde:issuedBy       employer:{employer_slug} ;")
    lines.append(f'  hrqde:sourceUrl      "{raw.source_uri}"^^xsd:anyURI ;')
    if req_uris:
        joined = " ,\n                       ".join(req_uris)
        lines.append(f"  hrqde:hasRequirement {joined} .")
    else:
        # keine Requirements: letztes Statement braucht den Punkt
        lines[-1] = lines[-1].rstrip(" ;") + " ."
    lines.append("")

    for req_uri, req in zip(req_uris, requirements):
        lines.append(req_uri)
        lines.append("  a                          hrqde:QualificationRequirement ;")
        lines.append(f"  hrqde:refersToCompetence   <{req.refers_to_competence}> ;")
        lines.append(f"  hrqde:requiredLevel        hrqde:{req.required_level} ;")
        lines.append(f'  hrqde:requirementKind      "{req.requirement_kind.value}" ;')
        lines.append(
            f'  hrqde:provenanceConfidence "{req.provenance_confidence:.2f}"^^xsd:decimal ;'
        )
        lines.append('  hrqde:provenanceSource     "job_posting" .')
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("persistence: %d Requirements -> %s", len(requirements), path)
    return path


def write_run_metadata(
    output_dir: Path,
    input_path: Path,
    outputs: list[Path],
    posting_count: int,
) -> Path:
    """Lauf-Parameter festhalten, damit ein Lauf reproduzierbar ist (Lieferspec §8)."""
    meta = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pipeline_version": __version__,
        "input": str(input_path),
        "posting_count": posting_count,
        "output_files": [str(p) for p in outputs],
        "parameters": {
            "spacy_model": SPACY_MODEL,
            "extractors": {
                "adj_noun": "spaCy nk/amod-Regel, deutscher Originaltext",
                "escoxlm_r": {
                    "skill_model": escoxlm_r.SKILL_MODEL,
                    "knowledge_model": escoxlm_r.KNOWLEDGE_MODEL,
                },
            },
            "esco_encoder": esco.DEFAULT_MODEL_ID,
            "esco_concepts": str(esco.DEFAULT_CONCEPTS_PATH),
            "mapping_threshold": MAPPING_THRESHOLD,
            "default_dqr_level": DEFAULT_DQR_LEVEL,
        },
    }
    path = output_dir / "run_metadata.json"
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("metadata: Lauf-Parameter -> %s", path)
    return path


def run(input_path: Path, output_dir: Path) -> list[Path]:
    raws = acquire(input_path)
    outputs: list[Path] = []
    for raw in raws:
        processed = process_nlp(raw)
        spans = extract_spans(raw, processed)
        draft = aggregate(raw, spans)
        mappings = map_to_esco(draft.span_candidates)
        requirements = build_requirements(draft.span_candidates, mappings)
        outputs.append(write_ttl(raw, requirements, output_dir))
    write_run_metadata(output_dir, input_path, outputs, len(raws))
    log.info("pipeline: fertig, %d TTL-Dateien geschrieben", len(outputs))
    return outputs
