"""Write results.md for one corpus from its hosted run logs: score in each run, median turns and tokens per question.
Usage: TALK2_CORPUS=pages_unseen uv run python -m talk2.results_table"""

import json
import statistics

from talk2.corpus_assembly import EVAL_ROOT
from talk2.scoring import load_questions

RUN_LABEL = "qwen3.5-9b-fp4-darkbloom"
MODEL_NOTE = "Qwen3.5-9B, 4-bit (fp4), Darkbloom through OpenRouter (`qwen/qwen3.5-9b`), temperature 0"
ROWS = [("PDF text, RAG, embedding", "rag"), ("PDF text, RAG, hybrid", "text_hybrid"), ("PDF text, grep, agentic search", "text_grep"), ("PDF text, BM25 in the loop", "text_bm25_loop"),
        ("RAG, embedding", "merged_rag"), ("RAG, embedding, thinking", "merged_rag_thinking"), ("RAG, hybrid", "merged_hybrid"),
        ("Grep, single fan-out", "merged_grep_fanout"), ("Grep, single fan-out, thinking", "merged_grep_fanout_thinking"),
        ("Grep agent", "merged_grep"), ("BM25 in the loop", "merged_bm25_loop"), ("BM25 in the loop + Thinking", "merged_bm25_loop_thinking"),
        ("Hybrid in the loop", "merged_fusion_loop"), ("Trace Fixes", "merged_grep_revision2"), ("Grep Token Reduction", "merged_grep_revision3"),
        ("Grep Token Reduction + Thinking", "merged_grep_revision3_thinking"),
        ("Grep Token Reduction, hint removed", "merged_grep_revision3_bare_empty_result"),
        ("BM25 Token Reduction", "merged_bm25_excerpt_fixed_count"), ("BM25 Token Reduction, model-selected count", "merged_bm25_excerpt_selected_count"),
        ("BM25 Token Reduction, first version", "merged_bm25_token_reduction"), ("BM25 Token Reduction, model-selected count, first version", "merged_bm25_selected_count")]


def complete_runs(configuration: str, question_count: int) -> dict[int, list[dict]]:
    runs = {}
    for path in sorted((EVAL_ROOT / "runs" / RUN_LABEL / configuration).glob("*.jsonl")):
        records = [json.loads(line) for line in path.open()]
        if len(records) == question_count and not any(record.get("error") for record in records):
            runs[records[0]["repeat"]] = records
    return runs


def write_results() -> None:
    question_count = len(load_questions())
    lines = [f"# Results: {EVAL_ROOT.name}", "", f"{question_count} questions. Model: {MODEL_NOTE}. Each run sends a different seed (41 + run number). Temperature 0 leaves the seed with little effect, so spread between runs most likely comes from the hosted endpoint.",
             "", "| Iteration | Configuration in the logs | Correct per run | Median turns | Median tokens |", "| --- | --- | ---: | ---: | ---: |"]
    for name, configuration in ROWS:
        runs = complete_runs(configuration, question_count)
        if not runs:
            continue
        records = [record for run in runs.values() for record in run]
        scores = ", ".join(str(sum(bool(r["score"]["correct"]) for r in runs[repeat])) for repeat in sorted(runs))
        turns = statistics.median(r.get("model_calls", 1) for r in records)
        tokens = statistics.median(r.get("prompt_tokens", 0) + r.get("completion_tokens", 0) for r in records)
        lines.append(f"| {name} | `{configuration}` | {scores} | {turns:g} | {tokens / 1000:.1f}k |")
    path = EVAL_ROOT / "results.md"
    existing_notes = path.read_text().partition("\n## Notes")[2] if path.exists() else ""
    notes = f"\n## Notes{existing_notes}" if existing_notes else ""
    path.write_text("\n".join(lines) + "\n" + notes)


if __name__ == "__main__":
    write_results()
