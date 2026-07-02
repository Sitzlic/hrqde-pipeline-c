"""Probelauf gegen die ESCOXLM-R-Modelle, nicht Teil der Pipeline.

Aufbau wie im offiziellen Demo-Space
(https://huggingface.co/spaces/jjzha/multilingual_skill_extraction).

Aufruf: python scripts/try_escoxlm_r.py [--input PATH]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import pipeline

SKILL_MODEL = "jjzha/escoxlmr_skill_extraction"
KNOWLEDGE_MODEL = "jjzha/escoxlmr_knowledge_extraction"
DEFAULT_INPUT = Path("data/input/example_posting.json")


def load_text(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    item = payload[0] if isinstance(payload, list) else payload
    return item["raw_text"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()

    text = load_text(args.input)
    print(f"Text ({len(text)} Zeichen): {text[:120]}...\n")

    skill = pipeline(model=SKILL_MODEL, aggregation_strategy="first")
    knowledge = pipeline(model=KNOWLEDGE_MODEL, aggregation_strategy="first")

    print("Skills:")
    for hit in skill(text):
        print(f"  {hit['score']:.2f}  {hit['word']:30s}  [{hit['start']}:{hit['end']}]")

    print("\nKnowledge:")
    for hit in knowledge(text):
        print(f"  {hit['score']:.2f}  {hit['word']:30s}  [{hit['start']}:{hit['end']}]")


if __name__ == "__main__":
    main()
