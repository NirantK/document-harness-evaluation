# Results: pages_multihop

20 questions. Model: Qwen3.5-9B, 4-bit (fp4), Darkbloom through OpenRouter (`qwen/qwen3.5-9b`), temperature 0. Each run sends a different seed (41 + run number). Temperature 0 leaves the seed with little effect, so spread between runs most likely comes from the hosted endpoint.

| Iteration | Configuration in the logs | Correct per run | Median turns | Median tokens |
| --- | --- | ---: | ---: | ---: |
| RAG, embedding | `merged_rag` | 9, 9, 9 | 1 | 2.5k |
| RAG, embedding, thinking | `merged_rag_thinking` | 9, 10, 9 | 1 | 4.1k |
| RAG, hybrid | `merged_hybrid` | 11, 11, 11 | 1 | 2.6k |
| Grep, single fan-out | `merged_grep_fanout` | 7, 7, 7 | 2 | 6.0k |
| Grep, single fan-out, thinking | `merged_grep_fanout_thinking` | 10, 6, 3 | 2 | 18.8k |
| Grep, agentic search | `merged_grep` | 13, 15, 13 | 12.5 | 68.0k |
| Trace Fixes | `merged_grep_revision2` | 14, 13, 13 | 12 | 109.4k |
| Token Reduction | `merged_grep_revision3` | 12, 14, 14 | 6.5 | 40.3k |
| Thinking | `merged_grep_revision3_thinking` | 16, 15, 16 | 8 | 44.4k |
| Token Reduction, hint removed | `merged_grep_revision3_bare_empty_result` | 13, 14, 11 | 8 | 48.4k |
