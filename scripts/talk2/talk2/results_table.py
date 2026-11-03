"""Write results.md for one corpus from its hosted run logs: score under each seed, median turns and tokens per question.
Usage: TALK2_CORPUS=pages_unseen uv run python -m talk2.results_table"""

import json
import statistics

from talk2.corpus_assembly import EVAL_ROOT
from talk2.scoring import load_questions

RUN_LABEL = "qwen3.5-9b-fp4-darkbloom"
MODEL_NOTE = "Qwen3.5-9B, 4-bit (fp4), Darkbloom through OpenRouter (`qwen/qwen3.5-9b`), temperature 0"
ROWS = [("RAG, embedding", "merged_rag"), ("RAG, embedding, thinking", "merged_rag_thinking"), ("RAG, hybrid", "merged_hybrid"),
        ("Grep, single fan-out", "merged_grep_fanout"), ("Grep, single fan-out, thinking", "merged_grep_fanout_thinking"),
        ("Grep, agentic search", "merged_grep"), ("Revision 2", "merged_grep_revision2"), ("Revision 3", "merged_grep_revision3"),
        ("Revision 3, thinking", "merged_grep_revision3_thinking")]


def complete_runs(configuration: str, question_count: int) -> dict[int, list[dict]]:
    runs = {}
    for path in sorted((EVAL_ROOT / "runs" / RUN_LABEL / configuration).glob("*.jsonl")):
        records = [json.loads(line) for line in path.open()]
        if len(records) == question_count and not any(record.get("error") for record in records):
            runs[records[0]["repeat"]] = records
    return runs


def write_results() -> None:
    question_count = len(load_questions())
    lines = [f"# Results: {EVAL_ROOT.name}", "", f"{question_count} questions. Model: {MODEL_NOTE}. Repeat N uses seed 41 + N.",
             "", "| Configuration | Correct per seed | Median turns | Median tokens |", "| --- | ---: | ---: | ---: |"]
    for name, configuration in ROWS:
        runs = complete_runs(configuration, question_count)
        if not runs:
            continue
        records = [record for run in runs.values() for record in run]
        scores = ", ".join(str(sum(bool(r["score"]["correct"]) for r in runs[repeat])) for repeat in sorted(runs))
        turns = statistics.median(r.get("model_calls", 1) for r in records)
        tokens = statistics.median(r.get("prompt_tokens", 0) + r.get("completion_tokens", 0) for r in records)
        lines.append(f"| {name} | {scores} | {turns:g} | {tokens / 1000:.1f}k |")
    (EVAL_ROOT / "results.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    write_results()
