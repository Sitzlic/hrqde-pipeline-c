from __future__ import annotations

import json
import logging
import uuid
from functools import lru_cache
from pathlib import Path

import spacy
from spacy.language import Language

from hrqde_c.extractors import escoxlm_r
from hrqde_c.mapping import esco
from hrqde_c.models import (
    JobPostingDraft,
    MappingDecision,
    ProcessedAdvertisement,
    QualificationRequirement,
    RawAdvertisement,
    SpanCandidate,
    SpanMapping,
    Token,
)

log = logging.getLogger(__name__)

SPACY_MODEL = "de_core_news_sm"
MAPPING_THRESHOLD = 0.5


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


def extract_spans(
    raw: RawAdvertisement, processed: ProcessedAdvertisement
) -> list[SpanCandidate]:
    # TODO regelbasierten ADJ+NOUN-Extraktor auf processed hinzufuegen (RC-C.1.1).
    spans = escoxlm_r.extract(raw)
    log.info("extraction: %s -> %d Span-Kandidaten", raw.id, len(spans))
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
    requirements = []
    for span in spans:
        mapping = by_span.get(span.id)
        if mapping is None:
            continue
        requirements.append(
            QualificationRequirement(
                id=f"req-{uuid.uuid4().hex[:8]}",
                refers_to_competence=mapping.concept_uri,
                required_level="DQR4",  # TODO DQR-Heuristik (RC-C.2.5)
                requirement_kind=span.requirement_kind,
                provenance_confidence=mapping.score,
            )
        )
    log.info("requirements: %d Spans -> %d Requirements", len(spans), len(requirements))
    return requirements


TTL_PREFIX = """@prefix hrqde: <http://hrqde.example/> .
@prefix esco:  <http://data.europa.eu/esco/skill/> .
@prefix xsd:   <http://www.w3.org/2001/XMLSchema#> .

"""


def write_ttl(
    raw: RawAdvertisement,
    requirements: list[QualificationRequirement],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{raw.id}.ttl"

    lines = [TTL_PREFIX]
    posting_uri = f"hrqde:posting/{raw.id}"
    lines.append(f"{posting_uri} a hrqde:JobPosting ;")
    lines.append(f'    hrqde:title "{raw.title}" ;')
    lines.append(f'    hrqde:employer "{raw.employer}" ;')
    lines.append(f'    hrqde:postingDate "{raw.posting_date}"^^xsd:date .')
    lines.append("")

    for req in requirements:
        req_uri = f"hrqde:requirement/{req.id}"
        lines.append(f"{req_uri} a hrqde:QualificationRequirement ;")
        lines.append(f"    hrqde:refersToCompetence <{req.refers_to_competence}> ;")
        lines.append(f'    hrqde:requiredLevel "{req.required_level}" ;')
        lines.append(f'    hrqde:requirementKind "{req.requirement_kind.value}" ;')
        lines.append(f"    hrqde:provenanceConfidence {req.provenance_confidence} .")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("persistence: %d Requirements -> %s", len(requirements), path)
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
    log.info("pipeline: fertig, %d TTL-Dateien geschrieben", len(outputs))
    return outputs
