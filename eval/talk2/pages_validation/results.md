# Results: pages_validation

20 questions. Model: Qwen3.5-9B, 4-bit (fp4), Darkbloom through OpenRouter (`qwen/qwen3.5-9b`), temperature 0. Each run sends a different seed (41 + run number). Temperature 0 leaves the seed with little effect, so spread between runs most likely comes from the hosted endpoint.

| Iteration | Configuration in the logs | Correct per run | Median turns | Median tokens |
| --- | --- | ---: | ---: | ---: |
| PDF text, RAG, embedding | `rag` | 7, 7, 7 | 1 | 2.5k |
| PDF text, RAG, hybrid | `text_hybrid` | 10, 10, 10 | 1 | 2.5k |
| PDF text, grep, agentic search | `text_grep` | 15, 17, 16 | 13 | 59.4k |
| RAG, embedding | `merged_rag` | 9, 9, 9 | 1 | 2.5k |
| RAG, embedding, thinking | `merged_rag_thinking` | 9, 7, 9 | 1 | 6.0k |
| RAG, hybrid | `merged_hybrid` | 11, 11, 11 | 1 | 2.5k |
| Grep, single fan-out | `merged_grep_fanout` | 6, 6, 7 | 2 | 7.5k |
| Grep, single fan-out, thinking | `merged_grep_fanout_thinking` | 7, 7, 6 | 2 | 21.5k |
| Grep, agentic search | `merged_grep` | 16, 17, 14 | 11 | 51.2k |
| Trace Fixes | `merged_grep_revision2` | 12, 12, 13 | 11.5 | 125.3k |
| Token Reduction | `merged_grep_revision3` | 12, 13, 15 | 7 | 46.9k |
| Thinking | `merged_grep_revision3_thinking` | 15, 18, 19 | 7 | 45.5k |
| Token Reduction, hint removed | `merged_grep_revision3_bare_empty_result` | 15, 15, 15 | 8 | 42.5k |
