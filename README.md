# Document Harness Evaluation

Code and method behind the talk **Better Harness, Better Answers: Document Intelligence Without Bigger Models**.

One small local model stays fixed. Every run uses `qwen3.5:9b` in Ollama at temperature 0 and seed 42. Only the harness
around it changes, and every change is scored on the same questions.

## The three components

| Component | What runs |
|---|---|
| Document processing | `maternion/LightOnOCR-3:0.8b` for pages with charts, images or no text; `pymupdf4llm` for the rest; PDF text-layer lines the conversion lost are merged back |
| Search | Embedding `qwen3-embedding:0.6b` (Qwen/Qwen3-Embedding-0.6B, 1024 dimensions) over LlamaIndex `SentenceSplitter(512, 50)` chunks in a FAISS `IndexFlatIP` (exact inner-product search), top 5; BM25 by `bm25s`; reciprocal rank fusion with k=60 over the top 20 of each, keeping 5; grep by ripgrep |
| Use of search | Linear pipe (retrieve, then answer once); single fan-out (one batch of search terms, then answer once); agentic loop with tools |

## Layout

- `scripts/talk2/talk2/`: the harness. `structured_agent.py` holds the agent prompts and revisions, `fanout_answering.py`
  the single fan-out query, `rag_answering.py` the linear pipe, `markdown_merging.py` the text-layer merge,
  `conversion_comparison.py` the conversion loss measurement.
- `eval/talk2/<corpus>/`: method notes, results and page selections for each question set. `pages_multihop` is the tuning
  set and `pages_unseen` the held-out set.
- `docs/files/`: selected source pages whose licences permit redistribution, served by GitHub Pages so a link ending in
  `#page=N` opens at that page. `docs/files/document_index.tsv` lists every document, its licence and its source.

Questions, per-page Markdown and answer logs are on Hugging Face:
[nirantk/document-harness-evaluation](https://huggingface.co/datasets/nirantk/document-harness-evaluation).

## Running

```bash
cd scripts/talk2
uv sync
TALK2_CORPUS=pages_multihop uv run talk2 run --configuration merged_grep_revision3 --repeats 1 --model qwen3.5:9b
uv run talk2 report
```

The XL-DocBench PDFs and question text are not redistributed. Fetch them from the XL-DocBench release, then build a corpus
with `uv run talk2 select-multihop`.

## Licence

Code and documentation: Apache License 2.0. Third-party documents and data keep their own terms; see `NOTICE`.
