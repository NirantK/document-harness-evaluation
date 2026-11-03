"""Run one configuration over the ten frozen questions in fixed order and log every answer as JSONL."""

import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
import time
from pathlib import Path

from talk2.corpus_assembly import CORPUS_NAME, CORPUS_PDF, EVAL_ROOT, REPOSITORY_ROOT
from talk2.model_client import (DEFAULT_MODEL, HOSTED_PROVIDER, SEED, HOSTED_QUANTIZATION, LOCAL_QUANTIZATION,
                                release_memory_if_short)
from talk2.scoring import load_questions, score

# configuration -> (input corpus, retrieval method). The first four are the frozen configurations; text_hybrid,
# markdown_hybrid and text_grep were added on 1 Nov to separate the effect of the input from that of retrieval.
CONFIGURATIONS = {
    "rag": ("text", "embedding"),
    "markdown_rag": ("markdown", "embedding"),
    "markdown_grep": ("markdown", "grep"),
    "markdown_grep_memory": ("markdown", "grep_memory"),
    "text_hybrid": ("text", "hybrid"),
    "markdown_hybrid": ("markdown", "hybrid"),
    "text_grep": ("text", "grep"),
    "markdown_grep_structured": ("markdown", "grep_structured"),
    "markdown_grep_revision2": ("markdown", "grep_revision2"),
    "markdown_grep_revision2_thinking": ("markdown", "grep_revision2_thinking"),
    "markdown_grep_fanout": ("markdown", "grep_fanout"),
    # The merged corpus restores text-layer lines that the converter lost; see markdown_merging.py.
    "merged_rag": ("markdown_merged", "embedding"),
    "merged_rag_thinking": ("markdown_merged", "embedding_thinking"),
    "merged_hybrid": ("markdown_merged", "hybrid"),
    "merged_grep_fanout": ("markdown_merged", "grep_fanout"),
    "merged_grep_fanout_thinking": ("markdown_merged", "grep_fanout_thinking"),
    "merged_grep": ("markdown_merged", "grep"),
    "merged_grep_revision2": ("markdown_merged", "grep_revision2"),
    "merged_grep_revision2_thinking": ("markdown_merged", "grep_revision2_thinking"),
    "merged_grep_revision3": ("markdown_merged", "grep_revision3"),
    "merged_grep_revision3_with_review": ("markdown_merged", "grep_revision3_with_review"),
    "merged_grep_revision3_thinking": ("markdown_merged", "grep_revision3_thinking"),
    "merged_grep_revision3_bare_empty_result": ("markdown_merged", "grep_revision3_bare_empty_result"),
    "merged_bm25_loop": ("markdown_merged", "bm25_loop"),
    "merged_fusion_loop": ("markdown_merged", "fusion_loop"),
    "text_grep_revision3": ("text", "grep_revision3"),
}
RUNS_ROOT = EVAL_ROOT / "runs"
CONVERTER_ID = "lightonocr-3-0.8b"
MODEL_LABELS = {DEFAULT_MODEL: "qwen3.6-35b-a3b", "gemma4:26b": "gemma4-26b-a4b", "qwen3.5:9b": "qwen3.5-9b",
                "gemma4:12b-it-q4_K_M": "gemma4-12b-q4"}


HOSTED_MODEL = "qwen/qwen3.5-9b"


def model_label(model: str) -> str:
    if model == HOSTED_MODEL:
        return f"qwen3.5-9b-{HOSTED_QUANTIZATION[HOSTED_PROVIDER]}-{HOSTED_PROVIDER.lower()}"
    return MODEL_LABELS.get(model) or re.sub(r"[^a-z0-9.]+", "-", model.lower().split("/")[-1]).strip("-")


def git_revision() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPOSITORY_ROOT,
                          capture_output=True, text=True).stdout.strip()


def completed_run(configuration: str, repeat: int, model: str) -> Path | None:
    """A log for this repeat with an answer to every question and no failed question, if one exists."""
    question_count = len(load_questions())
    for path in sorted((RUNS_ROOT / model_label(model) / configuration).glob(f"*-repeat{repeat}.jsonl")):
        records = [json.loads(line) for line in path.open()]
        if len(records) == question_count and not any(r.get("error") for r in records):
            return path
    return None


def run_configuration(configuration: str, repeat: int, model: str, backend: str, corpus_override: str = "",
                      workers: int = 1) -> Path:
    if configuration not in CONFIGURATIONS:
        raise ValueError(f"configuration must be one of {CONFIGURATIONS}")
    corpus, retrieval = CONFIGURATIONS[configuration]
    corpus = corpus_override or corpus
    questions = sorted(load_questions(), key=lambda q: q["order"])
    stamp = time.strftime("%Y%m%d-%H%M%S")
    label = configuration + (f"-on-{corpus}" if corpus_override else "")
    output = RUNS_ROOT / model_label(model) / label / f"{stamp}-repeat{repeat}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)

    if retrieval in ("embedding", "embedding_thinking", "hybrid"):
        from talk2.rag_answering import answer_question as rag_answer
        from talk2.retrieval_index import load_hybrid_retriever, load_retriever

        retriever = load_hybrid_retriever(corpus) if retrieval == "hybrid" else load_retriever(corpus)
    elif retrieval.startswith("grep_fanout"):
        from talk2.fanout_answering import answer_question as fanout_answer
        from talk2.structured_tools import StructuredCorpusTools

        structured_tools = StructuredCorpusTools(CORPUS_PDF.parent / corpus, ".md", show_long_lines=True)
    elif retrieval.startswith(("grep_structured", "grep_revision")):
        from talk2 import structured_agent
        from talk2.structured_agent import REVISION_1, REVISION_2, REVISION_2_THINKING, REVISION_3, REVISION_3_THINKING
        from talk2.structured_agent import answer_question as structured_answer
        from talk2.structured_tools import StructuredCorpusTools

        structured_tools = StructuredCorpusTools(CORPUS_PDF.parent / corpus, ".txt" if corpus == "text" else ".md",
                                                 show_long_lines=retrieval != "grep_structured")
        revision = {"grep_structured": REVISION_1, "grep_revision2": REVISION_2,
                    "grep_revision2_thinking": REVISION_2_THINKING, "grep_revision3": REVISION_3,
                    "grep_revision3_thinking": REVISION_3_THINKING,
                    "grep_revision3_with_review": structured_agent.REVISION_3_WITH_REVIEW,
                    "grep_revision3_bare_empty_result": structured_agent.REVISION_3_BARE_EMPTY_RESULT}[retrieval]
    else:
        from talk2.harness_agent import SessionMemory, answer_question as harness_answer
        from talk2.harness_tools import CorpusTools

        if retrieval in ("bm25_loop", "fusion_loop"):
            from talk2.ranked_search_tools import RankedCorpusTools

            tools = RankedCorpusTools(CORPUS_PDF.parent / corpus, ".md", corpus, use_embedding=retrieval == "fusion_loop")
        else:
            tools = CorpusTools(CORPUS_PDF.parent / corpus, ".txt" if corpus == "text" else ".md")
        memory = SessionMemory() if retrieval == "grep_memory" else None

    def answer(question: dict) -> dict:
        try:
            if retrieval in ("embedding", "embedding_thinking", "hybrid"):
                result = rag_answer(question["question"], retriever, model, backend,
                                    think=retrieval.endswith("_thinking"))
            elif retrieval.startswith("grep_fanout"):
                result = fanout_answer(question, structured_tools, model, backend,
                                       think=retrieval.endswith("_thinking"))
            elif retrieval.startswith(("grep_structured", "grep_revision")):
                result = structured_answer(question, structured_tools, model, backend, revision)
            else:
                result = harness_answer(question, tools, model, backend, memory)
            result["score"] = score(question, result["response"])
        except Exception as error:  # a failed question is logged as incorrect, the run continues
            result = {"error": f"{type(error).__name__}: {error}", "response": {},
                      "score": {"correct": False, "detail": "error"}}
        if backend == "ollama":
            release_memory_if_short(model)
        print(f"{configuration} r{repeat} #{question['order']} {question['id']}: "
              f"{'CORRECT' if result['score']['correct'] else 'wrong'} "
              f"answer={str(result['response'].get('answer', result.get('error', '')))[:90]!r}", flush=True)
        return result

    revision_id = git_revision()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(answer, questions))
    with output.open("w") as handle:
        for question, result in zip(questions, results):
            record = {"order": question["order"], "id": question["id"], "configuration": configuration,
                      "corpus": corpus, "retrieval": retrieval, "repeat": repeat, "model": model, "backend": backend,
                      "converter": {"markdown": CONVERTER_ID, "markdown_merged": f"{CONVERTER_ID}+text-layer-residue"}.get(
                          corpus, "pymupdf-get_text"),
                      "corpus_name": CORPUS_NAME, "git_revision": revision_id, "seed": SEED,
                      "quantization": HOSTED_QUANTIZATION[HOSTED_PROVIDER] if backend == "openrouter" else LOCAL_QUANTIZATION
                      if model == "qwen3.5:9b" else "", "hosted_provider": HOSTED_PROVIDER if backend == "openrouter" else "",
                      **result}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output
