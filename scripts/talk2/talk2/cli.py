"""Command-line entry point: one command per stage and per configuration."""

import time

import typer

from talk2.model_client import DEFAULT_MODEL

application = typer.Typer(no_args_is_help=True)


@application.callback()
def main() -> None:
    """Talk 2 evaluation commands."""

SMOKE_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the document corpus with a regular expression and return matching lines.",
        "parameters": {
            "type": "object",
            "properties": {"pattern": {"type": "string"}, "path_glob": {"type": "string"}},
            "required": ["pattern"],
        },
    },
}


@application.command()
def model_smoke_test(model: list[str], backend: str = "ollama") -> None:
    """Check that each model returns a well-formed tool call, then uses the tool result."""
    from talk2.model_client import chat

    for model_id in model:
        messages = [
            {"role": "system", "content": "Answer questions about the corpus. Use the search tool before answering."},
            {"role": "user", "content": "Which contract mentions $19,644,000?"},
        ]
        started = time.time()
        try:
            first = chat(messages, tools=[SMOKE_TOOL], model=model_id, backend=backend)
            call = first.tool_calls[0]
            messages += [
                {"role": "assistant", "content": first.content, "tool_calls": first.tool_calls},
                {"role": "tool", "tool_call_id": call["id"], "name": call["name"],
                 "content": "cuad/ADAPTIMMUNE_STRATEGIC_ALLIANCE.md:212: commit funding of at least ($19,644,000)"},
            ]
            second = chat(messages, tools=[SMOKE_TOOL], model=model_id, backend=backend)
            answer = second.content.strip().replace("\n", " ")
            typer.echo(f"PASS {model_id} {time.time() - started:.1f}s args={call['arguments']} "
                       f"tokens={second.prompt_tokens}/{second.completion_tokens} answer={answer[:120]}")
        except Exception as error:  # report and continue with the next model
            typer.echo(f"FAIL {model_id} {time.time() - started:.1f}s {type(error).__name__}: {str(error)[:200]}")


@application.command()
def assemble_corpus() -> None:
    """Symlink every corpus PDF into data/talk2_corpus/pdf and write eval/talk2/corpus_manifest.tsv."""
    from talk2.corpus_assembly import assemble

    rows = assemble()
    present = [row for row in rows if row["present"]]
    by_source: dict[str, list[int]] = {}
    for row in present:
        by_source.setdefault(row["source"], []).append(row["pages"])
    for source, pages in sorted(by_source.items()):
        typer.echo(f"{source:22s} files={len(pages):4d} pages={sum(pages):6d}")
    typer.echo(f"total files={len(present)} pages={sum(row['pages'] for row in present)}")
    for row in rows:
        if not row["present"]:
            typer.echo(f"ABSENT {row['source']} {row['doc_id']} {row['title']}")


@application.command()
def extract_text() -> None:
    """Baseline PyMuPDF text for every corpus PDF into data/talk2_corpus/text (skips files already extracted)."""
    from talk2.text_extraction import extract_all

    documents, pages = extract_all()
    typer.echo(f"extracted documents={documents} pages={pages}")


@application.command()
def score_self_test() -> None:
    """Gold-shaped answers must score correct and known wrong answers incorrect."""
    from talk2.scoring import self_test

    failures = self_test()
    for failure in failures:
        typer.echo(f"FAIL {failure}")
    typer.echo("self-test passed" if not failures else f"{len(failures)} failures")
    raise typer.Exit(1 if failures else 0)


@application.command()
def build_index(corpus: str = typer.Option("text", help="text (PyMuPDF baseline) or markdown")) -> None:
    """Chunk, embed and store one corpus in a FAISS index under eval/talk2/cache."""
    from talk2.retrieval_index import build_index as build

    typer.echo(f"indexed chunks={build(corpus)} corpus={corpus}")


@application.command()
def run(configuration: str = typer.Option(..., help="rag | markdown_rag | markdown_grep | markdown_grep_memory | text_hybrid | markdown_hybrid | text_grep"),
        repeats: int = 3, first_repeat: int = 1, model: str = DEFAULT_MODEL, backend: str = "ollama",
        resume: bool = typer.Option(False, help="Skip a repeat that already has a complete log"),
        corpus_override: str = typer.Option("", help="Development only: run a configuration on another corpus"),
        workers: int = typer.Option(1, help="Questions answered at once; above 1 only with a hosted backend")) -> None:
    """Answer the ten frozen questions in fixed order; one JSONL log per repeat under eval/talk2/runs."""
    from talk2.evaluation_run import completed_run, run_configuration

    for repeat in range(first_repeat, first_repeat + repeats):
        existing = completed_run(configuration, repeat, model) if resume and not corpus_override else None
        if existing:
            typer.echo(f"already complete {existing}")
            continue
        typer.echo(f"wrote {run_configuration(configuration, repeat, model, backend, corpus_override, workers)}")


@application.command()
def convert_markdown(workers: int = 2, source: list[str] = typer.Option(None, help="Limit to these sources"),
                     page_limit: int = typer.Option(0, help="Pilot: convert only this many pending pages"),
                     only_route: str = typer.Option("", help="converter or pymupdf4llm: process only that route")) -> None:
    """Route and convert every pending page to Markdown (resumable), then assemble per-document Markdown files."""
    import asyncio

    from talk2.markdown_conversion import assemble_documents, convert_pages, pending_pages

    pages = pending_pages(set(source) if source else None)
    if page_limit:
        pages = pages[:page_limit]
    typer.echo(f"pending pages={len(pages)} workers={workers}")
    asyncio.run(convert_pages(pages, workers, only_route=only_route))
    documents, missing = assemble_documents()
    typer.echo(f"assembled documents={documents} pages still missing={missing}")


@application.command()
def report(repeats: int = 3) -> None:
    """Write eval/talk2/results.md and review.tsv from the newest run logs."""
    from talk2.results_report import build_report

    typer.echo(build_report(repeats))


@application.command()
def select_pages() -> None:
    """Mine the pages100 corpus (evidence + embedding and BM25 hard negatives) and package it."""
    from collections import Counter

    from talk2.page_selection import package, select

    selection = select()
    package(selection)
    typer.echo(f"selected pages={len(selection)} documents={len({r['document'] for r in selection})}")
    for (question, role), count in sorted(Counter((r["question"], r["role"]) for r in selection).items()):
        typer.echo(f"  {question:40s} {role:20s} {count}")


@application.command()
def select_multihop() -> None:
    """Select the multi-hop questions by the fixed rule and package the pages_multihop corpus."""
    from talk2.multihop_selection import select_and_package

    typer.echo(select_and_package())

@application.command("assemble-merged")
def assemble_merged_command() -> None:
    """Write the markdown_merged corpus from the converted pages and the PDF text layer."""
    from talk2.markdown_conversion import assemble_merged_documents

    documents, missing, residue = assemble_merged_documents()
    typer.echo(f"merged documents={documents} missing_pages={missing} pages_with_residue={residue}")


@application.command("export-dataset")
def export_dataset_command(corpus: str = "pages_holdout") -> None:
    """Write the local Hugging Face dataset, full and public variants, from the largest built corpus."""
    from talk2.dataset_export import export_dataset

    print(export_dataset(corpus))


@application.command("publish-dataset")
def publish_dataset_command() -> None:
    """Upload the public variant. Reads the token from HF_TOKEN."""
    import os

    from talk2.dataset_export import publish_public_variant

    print(publish_public_variant(os.environ["HF_TOKEN"]))


if __name__ == "__main__":
    application()
