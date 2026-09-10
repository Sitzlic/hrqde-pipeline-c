from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class SpanType(str, Enum):
    SKILL = "skill"
    KNOWLEDGE = "knowledge"


class RequirementKind(str, Enum):
    MUST = "must"
    NICE_TO_HAVE = "nice_to_have"


class MappingDecision(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class RawAdvertisement(BaseModel):
    # Provenance-Felder sind Pflicht (SA-C.1.04)
    id: str
    source_uri: str
    crawl_date: str
    platform: str
    employer: str
    posting_date: str
    title: str
    raw_text: str
    # ISCO-Hauptgruppe fuer den DQR-Default, fehlt sie greift der Fallback
    isco_group: int | None = None


class Token(BaseModel):
    text: str
    pos: str
    lemma: str


class ProcessedAdvertisement(BaseModel):
    raw_id: str
    language: str
    tokens: list[Token]


class SpanCandidate(BaseModel):
    id: str
    processed_raw_id: str
    text: str
    span_type: SpanType
    char_start: int
    char_end: int
    extractor: Literal["adj_noun", "escoxlm_r", "adj_noun_en"]
    requirement_kind: RequirementKind = RequirementKind.MUST
    score: float | None = None


class JobPostingDraft(BaseModel):
    id: str
    raw_advertisement: RawAdvertisement
    span_candidates: list[SpanCandidate]


class EscoConcept(BaseModel):
    uri: str
    pref_label: str
    alt_labels: list[str] = Field(default_factory=list)
    definition: str | None = None


class EscoOccupation(BaseModel):
    uri: str
    pref_label: str
    # ESCO-Code ist ISCO-08-basiert, erste Ziffer = Hauptgruppe
    code: str
    isco_major_group: int
    essential_skills: list[str] = Field(default_factory=list)


class SpanMapping(BaseModel):
    id: str
    span_id: str
    concept_uri: str
    score: float
    decision: MappingDecision
    # 1 = bester Treffer unter den Top-k
    rank: int = 1
    # ueber welches Konzept-Label (pref oder alt, DE oder EN) der Treffer kam
    matched_label: str = ""


class QualificationRequirement(BaseModel):
    id: str
    refers_to_competence: str
    required_level: str
    requirement_kind: RequirementKind
    provenance_confidence: float


class GoldSpan(BaseModel):
    text: str
    span_type: SpanType
    # Positionen im Token-Bestand der Anzeige, Ende exklusiv
    token_start: int
    token_end: int
    # optionale manuell zugeordnete Referenz-URI fuer die Mapping-Bewertung
    esco_uri: str | None = None


class EvaluationRun(BaseModel):
    """Ein Auswertungslauf einer Extraktionsvariante gegen den Testteil."""

    id: str
    variant: str
    gold_dataset: str
    partition: Literal["kalibrier", "test"] = "test"
    advertisement_count: int
    run_timestamp: str


class EvaluationResult(BaseModel):
    run_id: str
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int


class GoldAdvertisement(BaseModel):
    """Annotierte Variante einer RawAdvertisement (BIO-Schema, SA-C.3.02)."""

    raw_id: str
    # Kalibrierteil tunt den Threshold, Testteil misst (SA-C.3.06)
    partition: Literal["kalibrier", "test"]
    tokens: list[str]
    bio_labels: list[str]
    spans: list[GoldSpan]
    annotator_ids: list[str]
    # Cohen's Kappa, nur bei doppelt annotierten Anzeigen belegt (SA-C.3.03)
    inter_annotator_agreement: float | None = None
