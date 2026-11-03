# Results: pages_unseen

20 questions. Model: Qwen3.5-9B, 4-bit (fp4), Darkbloom through OpenRouter (`qwen/qwen3.5-9b`), temperature 0. Repeat N uses seed 41 + N.

| Configuration | Correct per seed | Median turns | Median tokens |
| --- | ---: | ---: | ---: |
| RAG, embedding | 5, 5, 5 | 1 | 2.6k |
| RAG, embedding, thinking | 8, 7, 5 | 1 | 4.5k |
| RAG, hybrid | 7, 7, 6 | 1 | 2.7k |
| Grep, single fan-out | 5, 5, 4 | 2 | 6.6k |
| Grep, single fan-out, thinking | 4, 4, 2 | 2 | 20.5k |
| Grep, agentic search | 7, 9, 7 | 20 | 155.3k |
| Revision 2 | 7, 6, 7 | 13 | 138.2k |
| Revision 3 | 8, 7, 8 | 13 | 98.1k |
| Revision 3, thinking | 11, 8, 8 | 11 | 94.5k |

## Notes

- All 20 questions come from XL-DocBench; 12 of the 20 tuning questions do. Revision 3 scores 6 to 8 of those 12 and 7 to 8 of the other 8.
- Revision 3 was frozen under the tag `talk2-revision3-frozen` before its first run here.
- A first run of revision 3 with thinking on, seed 42, at a 4,096-token output limit scored 13. Thinking shares that limit, and it
  left the fan-out with no search terms, so thinking runs moved to 16,384 tokens. Those are the scores above.
- Scoring is by alias, the same for every row. On seed 42 two of revision 3's correct answers, questions 17 and 20, also list an
  extra item. A stricter rule would mark them wrong.
- Local check on the tuning set, revision 3, Qwen3.5-9B nvfp4 in Ollama: 17 of 20, median 6 turns, 41k tokens, 16.8 minutes.
