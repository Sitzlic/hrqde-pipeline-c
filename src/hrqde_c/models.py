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
    extractor: Literal["adj_noun", "escoxlm_r"]
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


class SpanMapping(BaseModel):
    id: str
    span_id: str
    concept_uri: str
    score: float
    decision: MappingDecision
    # 1 = bester Treffer unter den Top-k
    rank: int = 1


class QualificationRequirement(BaseModel):
    id: str
    refers_to_competence: str
    required_level: str
    requirement_kind: RequirementKind
    provenance_confidence: float
