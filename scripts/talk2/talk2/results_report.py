"""Build results.md and review.tsv from the run logs, for every answer model under runs/<model>/<configuration>/.

Every logged answer is re-scored with the current rules in scoring.py, so a rule decision applies to all runs alike.
"""

import csv
import json
import statistics
from collections import defaultdict

from talk2.corpus_assembly import REPOSITORY_ROOT
from talk2.evaluation_run import CONFIGURATIONS, MODEL_LABELS, RUNS_ROOT
from talk2.scoring import load_questions, score

RESULTS = RUNS_ROOT.parent / "results.md"
REVIEW = RUNS_ROOT.parent / "review.tsv"
NOTES = RUNS_ROOT.parent / "results_notes.md"
DISPLAY_ORDER = ("rag", "text_hybrid", "text_grep", "markdown_rag", "markdown_hybrid", "markdown_grep",
                 "markdown_grep_memory", "markdown_grep_structured", "markdown_grep_revision2",
                 "markdown_grep_revision2_thinking")
INPUT_TITLES = {"text": "PDF text layer", "markdown": "Markdown (DSH step 1)"}
RETRIEVAL_TITLES = {"embedding": "Embedding top-5", "hybrid": "BM25 + embedding top-5", "grep": "Grep agent",
                    "grep_memory": "Grep agent + translate, memory",
                    "grep_structured": "Structured grep agent",
                    "grep_revision2": "Structured grep agent, revision 2",
                    "grep_revision2_thinking": "Revision 2 with thinking on"}
SHORT_TITLES = {"rag": "Text · emb", "text_hybrid": "Text · hybrid", "text_grep": "Text · grep",
                "markdown_rag": "MD · emb", "markdown_hybrid": "MD · hybrid", "markdown_grep": "MD · grep",
                "markdown_grep_memory": "MD · grep+mem", "markdown_grep_structured": "MD · structured",
                "markdown_grep_revision2": "MD · rev 2", "markdown_grep_revision2_thinking": "MD · rev 2 think"}


def latest_repeats(model: str, configuration: str, repeats: int, question_count: int) -> list[list[dict]]:
    """The newest complete log per repeat number, so a re-run of repeat 2 replaces the older repeat 2."""
    by_repeat: dict[int, list] = {}
    for path in sorted((RUNS_ROOT / model / configuration).glob("*.jsonl")):
        records = [json.loads(line) for line in path.open()]
        if len(records) == question_count:
            by_repeat[records[0]["repeat"]] = records
    return [by_repeat[k] for k in sorted(by_repeat)][:repeats]


def model_directories() -> list[str]:
    present = [p.name for p in RUNS_ROOT.iterdir() if p.is_dir()] if RUNS_ROOT.exists() else []
    known = [label for label in MODEL_LABELS.values() if label in present]
    return known + sorted(set(present) - set(known))


def build_report(repeats: int = 3) -> str:
    questions = sorted(load_questions(), key=lambda q: q["order"])
    by_id = {q["id"]: q for q in questions}
    table = defaultdict(dict)    # (model, configuration) -> {question id: (correct, count)}
    totals = {}                  # (model, configuration) -> per-repeat totals
    review_rows = []
    models = model_directories()
    for model in models:
        for configuration in DISPLAY_ORDER:
            runs = latest_repeats(model, configuration, repeats, len(questions))
            if not runs:
                continue
            for run in runs:
                for record in run:
                    record["score"] = score(by_id[record["id"]], record.get("response") or {})
            totals[model, configuration] = [sum(r["score"]["correct"] for r in run) for run in runs]
            for question in questions:
                outcomes = [next(r for r in run if r["id"] == question["id"]) for run in runs]
                correct = sum(o["score"]["correct"] for o in outcomes)
                table[model, configuration][question["id"]] = (correct, len(outcomes))
                if 0 < correct < len(outcomes) or any((o.get("response") or {}).get("parse_error") or o.get("error")
                                                      for o in outcomes):
                    for o in outcomes:
                        response = o.get("response") or {}
                        review_rows.append({"model": model, "configuration": configuration, "repeat": o["repeat"],
                                            "id": o["id"], "auto_correct": o["score"]["correct"],
                                            "detail": o["score"]["detail"],
                                            "answer": str(response.get("answer", o.get("error", "")))[:300],
                                            "files": "; ".join(response.get("files", []))})

    def majority(model: str, configuration: str, question_set: str = "") -> int:
        cells = table[model, configuration]
        return sum(1 for question_id, (correct, count) in cells.items() if correct * 2 > count
                   and (not question_set or by_id[question_id].get("set") == question_set))

    def total_cell(model: str, configuration: str, question_set: str = "") -> str:
        if (model, configuration) not in totals:
            return "–"
        values = totals[model, configuration]
        spread = "" if question_set or min(values) == max(values) else f" ({min(values)}–{max(values)})"
        return f"**{majority(model, configuration, question_set)}**{spread}"

    def score_table(heading: str, question_set: str = "") -> list[str]:
        count = sum(1 for q in questions if not question_set or q.get("set") == question_set)
        rows = ["", f"## {heading}, out of {count}", "",
                "| Input | Retrieval | Configuration | " + " | ".join(models) + " |",
                "|---|---|---|" + "---|" * len(models)]
        for configuration in DISPLAY_ORDER:
            corpus, retrieval = CONFIGURATIONS[configuration]
            rows.append(f"| {INPUT_TITLES[corpus]} | {RETRIEVAL_TITLES[retrieval]} | `{configuration}` | "
                        + " | ".join(total_cell(m, configuration, question_set) for m in models) + " |")
        return rows

    lines = ["# Talk 2 results: plain RAG vs the Deep Seek Harness", "",
             f"Generated by `uv run talk2 report` from `{RUNS_ROOT.relative_to(REPOSITORY_ROOT)}/`. Method: "
             f"`method.md`. A question counts as right when the majority of its repeats is correct; a total is the "
             f"number of questions right out of {len(questions)}, with the per-repeat range when repeats differ."]
    lines += score_table("All questions")
    if any(q.get("set") == "multi_hop" for q in questions):
        lines += score_table("The original ten (rag misses)", "rag_misses")
        lines += score_table("The ten multi-hop questions", "multi_hop")
    for model in models:
        columns = [c for c in DISPLAY_ORDER if (model, c) in totals]
        repeat_counts = sorted({len(totals[model, c]) for c in columns})
        lines += ["", f"## {model}: per question", "",
                  f"Cells: correct repeats out of {'/'.join(map(str, repeat_counts))}.", "",
                  "| # | Question | " + " | ".join(SHORT_TITLES[c] for c in columns) + " |",
                  "|---|---|" + "---|" * len(columns)]
        for question in questions:
            cells = []
            for c in columns:
                correct, count = table[model, c].get(question["id"], (0, 0))
                cells.append(f"{'✅' if count and correct * 2 > count else '❌'} {correct}/{count}")
            lines.append(f"| {question['order']} | {question['id']} | " + " | ".join(cells) + " |")
        lines.append("| | **Total** | " + " | ".join(
            f"**{majority(model, c)}** (median {statistics.median(totals[model, c]):g})" for c in columns) + " |")
    lines += ["", f"{len(review_rows)} rows in `review.tsv` (repeats that disagree, unparseable replies or errors)."]
    if NOTES.exists():
        lines += ["", NOTES.read_text().strip()]
    with REVIEW.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "configuration", "repeat", "id", "auto_correct",
                                                    "detail", "answer", "files"], delimiter="\t")
        writer.writeheader()
        writer.writerows(review_rows)
    RESULTS.write_text("\n".join(lines) + "\n")
    return "\n".join(lines)
