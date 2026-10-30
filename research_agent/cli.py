"""Command-line interface.

    python -m research_agent "How do vector databases compare for RAG workloads?"
    python -m research_agent "..." --yes --out report.md     # skip plan approval
    python -m research_agent --list
    python -m research_agent --resume <job_id>
    python -m research_agent --show <job_id>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from research_agent.config import get_settings
from research_agent.db import init_db
from research_agent.logging import configure_logging
from research_agent.service import ResearchService

console = Console()


def show_plan(plan: dict) -> None:
    lines = [f"[bold]Objective:[/bold] {plan['objective']}", f"[bold]Scope:[/bold] {plan['scope']}", ""]
    for sq in plan["subquestions"]:
        lines.append(f"[bold cyan]{sq['id']}[/bold cyan] {sq['question']}")
        for q in sq["search_queries"]:
            lines.append(f"     [dim]search:[/dim] {q}")
    console.print(Panel("\n".join(lines), title="Research plan", expand=False))


def edit_plan(plan: dict) -> dict:
    console.print("Enter replacement subquestions, one per line; a blank line finishes. "
                  "Each new subquestion is also used as its search query.")
    subqs = []
    while line := input(f"  Q{len(subqs) + 1}: ").strip():
        subqs.append({"id": f"Q{len(subqs) + 1}", "question": line, "search_queries": [line]})
    if subqs:
        plan = {**plan, "subquestions": subqs, "outline": [q["question"] for q in subqs]}
    return plan


def show_metrics(job: dict) -> None:
    ev = job.get("evaluation") or {}
    table = Table(title="Run metrics", show_header=False)
    rows = [
        ("Citation coverage", f"{ev.get('citation_coverage', 0):.0%}"),
        ("Hallucination risk", f"{ev.get('hallucination_risk', 0):.0%}"),
        ("Sources cited / unique domains", f"{ev.get('cited_sources', 0)} / {ev.get('unique_domains', 0)}"),
        ("Claims verified / uncertain / contradicted",
         f"{ev.get('claims_verified', 0)} / {ev.get('claims_uncertain', 0)} / {ev.get('claims_contradicted', 0)}"),
        ("Injection attempts removed", str(ev.get("injection_flags", 0))),
        ("Latency", f"{ev.get('latency_s', 0):.1f}s"),
        ("LLM calls / tokens", f"{ev.get('llm_calls', 0)} / {ev.get('tokens', 0):,}"),
        ("Cost", f"${job.get('cost_usd', 0):.4f}"),
    ]
    for k, v in rows:
        table.add_row(k, v)
    console.print(table)
    timeline = Table(title="Timeline")
    for col in ("node", "duration", "ok", "cost"):
        timeline.add_column(col)
    for e in job["timeline"]:
        timeline.add_row(e["node"], f"{e['duration_ms'] / 1000:.1f}s", "yes" if e["success"] else "NO",
                         f"${e['cost_usd']:.4f}")
    console.print(timeline)


def drive(svc: ResearchService, job_id: str, status: str) -> str:
    while status == "awaiting_approval":
        show_plan(svc.get_job(job_id)["plan"])
        choice = Prompt.ask("Approve this plan?", choices=["y", "e", "n"], default="y")
        plan = svc.get_job(job_id)["plan"]
        if choice == "e":
            plan = edit_plan(plan)
        with console.status("Researching..."):
            status = svc.approve(job_id, choice != "n", plan if choice == "e" else None, background=False)
    return status


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="research_agent", description="Autonomous research agent")
    p.add_argument("question", nargs="?")
    p.add_argument("-y", "--yes", action="store_true", help="auto-approve the research plan")
    p.add_argument("-o", "--out", type=Path, help="write the report to this file")
    p.add_argument("--list", action="store_true", help="list recent jobs")
    p.add_argument("--resume", metavar="JOB_ID", help="retry a failed job from its last checkpoint")
    p.add_argument("--show", metavar="JOB_ID", help="print a finished job's report and metrics")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    settings = get_settings()
    configure_logging("INFO" if args.verbose else "WARNING", settings.log_json)
    init_db()
    svc = ResearchService(settings)

    if args.list:
        table = Table("job id", "status", "cost", "question")
        for j in svc.list_jobs():
            table.add_row(j["job_id"], j["status"], f"${j['cost_usd']:.4f}", j["question"][:70])
        console.print(table)
        return 0

    if args.show:
        job_id = args.show
        status = svc.get_job(job_id)["status"]
    elif args.resume:
        job_id = args.resume
        with console.status("Resuming..."):
            status = svc.resume(job_id, background=False)
    elif args.question:
        job_id = svc.create_job(args.question, auto_approve=args.yes)
        console.print(f"[dim]job {job_id} · {settings.provider}/{settings.model}[/dim]")
        with console.status("Planning..." if not args.yes else "Researching..."):
            status = svc.run(job_id)
    else:
        p.print_help()
        return 1

    status = drive(svc, job_id, status)
    job = svc.get_job(job_id)
    if status != "completed":
        console.print(f"[bold red]Job {status}[/bold red]: {job.get('error') or ''}")
        if status == "failed":
            console.print(f"Retry from the last checkpoint with: python -m research_agent --resume {job_id}")
        return 2

    report = svc.get_job_row(job_id).report
    console.print(Markdown(report))
    show_metrics(job)
    if args.out:
        args.out.write_text(report, encoding="utf-8")
        console.print(f"Report written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
