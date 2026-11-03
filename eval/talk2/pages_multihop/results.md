# Results: pages_multihop

20 questions. Model: Qwen3.5-9B, 4-bit (fp4), Darkbloom through OpenRouter (`qwen/qwen3.5-9b`), temperature 0. Repeat N uses seed 41 + N.

| Configuration | Correct per seed | Median turns | Median tokens |
| --- | ---: | ---: | ---: |
| RAG, embedding | 9, 9, 9 | 1 | 2.5k |
| RAG, embedding, thinking | 9, 10, 9 | 1 | 4.1k |
| RAG, hybrid | 11, 11, 11 | 1 | 2.6k |
| Grep, single fan-out | 7, 7, 7 | 2 | 6.0k |
| Grep, single fan-out, thinking | 10, 6, 3 | 2 | 18.8k |
| Grep, agentic search | 13, 15, 13 | 12.5 | 68.0k |
| Revision 2 | 14, 13, 13 | 12 | 109.4k |
| Revision 3 | 12, 14, 14 | 6.5 | 40.3k |
| Revision 3, thinking | 16, 15, 16 | 8 | 44.4k |
