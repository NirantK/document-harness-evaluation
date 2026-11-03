# Document Harness Evaluation

Code and method behind the talk **Better Harness, Better Answers: Document Intelligence Without Bigger Models**.

One small model stays fixed: Qwen3.5-9B at temperature 0. Final runs use the 4-bit (fp4) weights on Darkbloom through OpenRouter,
`qwen/qwen3.5-9b`, three runs. Earlier runs use the nvfp4 weights in Ollama, `qwen3.5:9b`, one run. Every run log names its
model, quantization, provider and the seed sent; at temperature 0 the seed has little effect.

The talk displays its iterations under these names: Trace Fixes is configuration `merged_grep_revision2`, Token Reduction is
`merged_grep_revision3`, and Thinking is `merged_grep_revision3_thinking`. Only the harness around the model changes, and every change is scored on the same
questions. Embeddings and page conversion stay local in Ollama.

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
- `eval/talk2/<corpus>/`: method notes, results and page selections for each question set. Validation is `pages_validation`
  and test is `pages_unseen`; both search the same 531 pages. `pages_multihop` holds earlier validation runs on 222 of them.
- `docs/files/`: selected source pages whose licences permit redistribution, served by GitHub Pages so a link ending in
  `#page=N` opens at that page. `docs/files/document_index.tsv` lists every document, its licence and its source.

Questions, per-page Markdown and answer logs are on Hugging Face:
[nirantk/document-harness-evaluation](https://huggingface.co/datasets/nirantk/document-harness-evaluation).

## Running

```bash
cd scripts/talk2
uv sync
# Hosted, fp4 on Darkbloom, 20 questions in about two minutes; needs OPENROUTER_API_KEY
TALK2_CORPUS=pages_multihop uv run talk2 run --configuration merged_grep_revision3 --repeats 1 --model qwen/qwen3.5-9b --backend openrouter --workers 10
# Local, nvfp4 in Ollama, about 40 minutes
TALK2_CORPUS=pages_multihop uv run talk2 run --configuration merged_grep_revision3 --repeats 1 --model qwen3.5:9b
uv run talk2 report
```

The XL-DocBench PDFs and question text are not redistributed. Fetch them from the XL-DocBench release, then build a corpus
with `uv run talk2 select-multihop`.

## Licence

Code and documentation: Apache License 2.0. Third-party documents and data keep their own terms; see `NOTICE`.
