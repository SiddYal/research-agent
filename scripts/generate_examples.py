"""Runs the agent on the reference questions and saves each report + metrics to examples/.

    python scripts/generate_examples.py            # all questions
    python scripts/generate_examples.py 2          # just the second one
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research_agent.config import get_settings  # noqa: E402
from research_agent.db import init_db  # noqa: E402
from research_agent.logging import configure_logging  # noqa: E402
from research_agent.service import ResearchService  # noqa: E402

QUESTIONS = [
    "How do pgvector, Pinecone, and Weaviate compare for retrieval-augmented generation workloads?",
    "How do LangGraph, CrewAI, and AutoGen differ as frameworks for building LLM agents?",
    "What is driving the growth in AI data center power demand, and how are operators responding?",
]


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60]


def main() -> int:
    configure_logging("INFO")
    init_db()
    svc = ResearchService(get_settings())
    picked = [QUESTIONS[int(a) - 1] for a in sys.argv[1:]] or QUESTIONS
    out = ROOT / "examples"
    out.mkdir(exist_ok=True)
    failures = 0
    for q in picked:
        job_id = svc.create_job(q, auto_approve=True)
        status = svc.run(job_id)
        job = svc.get_job(job_id)
        if status != "completed":
            print(f"FAILED ({status}): {q}\n  {job['error']}")
            failures += 1
            continue
        name = slug(q)
        (out / f"{name}.md").write_text(svc.get_job_row(job_id).report, encoding="utf-8")
        (out / f"{name}.metrics.json").write_text(json.dumps(
            {"question": q, "model": job["model"], "cost_usd": job["cost_usd"],
             "evaluation": job["evaluation"], "timeline": job["timeline"]}, indent=2, default=str),
            encoding="utf-8")
        ev = job["evaluation"]
        print(f"OK  {name}: coverage {ev['citation_coverage']:.0%}, hallucination risk "
              f"{ev['hallucination_risk']:.0%}, {ev['claims_verified']}/{ev['claims_total']} claims verified, "
              f"${job['cost_usd']:.4f}, {ev['latency_s']:.0f}s")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
