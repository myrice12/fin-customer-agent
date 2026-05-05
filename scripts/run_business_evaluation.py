"""
Run offline business metric evaluation and write a Markdown report.

Usage:
    python scripts/run_business_evaluation.py
    python scripts/run_business_evaluation.py --output reports/business_eval.md
    python scripts/run_business_evaluation.py --routing-mode skip
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.evaluation.business_metrics import render_business_report, run_business_evaluation


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Run business metric evaluation.")
    parser.add_argument("--kb-dir", default="knowledge_base", help="Knowledge base directory.")
    parser.add_argument(
        "--output",
        default="business_evaluation_report.md",
        help="Markdown report output path.",
    )
    parser.add_argument(
        "--routing-mode",
        choices=["live", "skip"],
        default="live",
        help="Use live Supervisor routing or skip routing when no LLM API is available.",
    )
    args = parser.parse_args()

    report = await run_business_evaluation(kb_dir=args.kb_dir, routing_mode=args.routing_mode)
    markdown = render_business_report(report)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")

    summary = report["summary"]
    print("Business evaluation completed")
    print(f"Report: {output_path}")
    print(f"RAG Recall@5: {summary['recall_at_5']:.4f}")
    print(f"RAG MRR@5: {summary['mrr_at_5']:.4f}")
    if summary["auto_routing_accuracy"] is None:
        print("Auto routing accuracy: not evaluated")
    else:
        print(f"Auto routing accuracy: {summary['auto_routing_accuracy']:.4f}")
    print(f"Violation recall: {summary['violation_recall']:.4f}")
    print(f"Explicit violation recall: {summary['explicit_violation_recall']:.4f}")
    print(f"Implicit violation recall: {summary['implicit_violation_recall']:.4f}")
    print(f"False positive rate: {summary['false_positive_rate']:.4f}")
    print(f"Context compression ratio: {summary['context_compression_ratio']:.4f}")


if __name__ == "__main__":
    asyncio.run(_main())
