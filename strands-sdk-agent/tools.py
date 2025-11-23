from typing import List
from pydantic import BaseModel
from strands_agents_tools import tool


# ---- Tool Input & Output Schemas ----

class ExtractBulletsInput(BaseModel):
    text: str

class ExtractBulletsOutput(BaseModel):
    bullets: List[str]


class FinalizeSummaryInput(BaseModel):
    bullets: List[str]

class FinalizeSummaryOutput(BaseModel):
    summary: str


# ---- Actual Tool Functions (LLM calls these deterministically) ----
@tool
def extract_bullets(args: ExtractBulletsInput) -> ExtractBulletsOutput:
    # in real life you’d do actual NLP or LLM inside
    # here we simulate deterministic “bullet extraction”
    lines = [l.strip("-• ") for l in args.text.split("\n") if len(l.strip()) > 0]
    return ExtractBulletsOutput(bullets=lines)

@tool
def finalize_summary(args: FinalizeSummaryInput) -> FinalizeSummaryOutput:
    # join bullets in a structured form
    summary = "Here is the structured summary:\n" + "\n".join(
        f"- {b}" for b in args.bullets
    )
    return FinalizeSummaryOutput(summary=summary)