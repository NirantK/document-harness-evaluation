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

- All 20 questions come from XL-DocBench; 12 of the 20 tuning questions do.
- Revision 3 was frozen under the tag `talk2-revision3-frozen` before its first run here. No run in this table replayed a saved reply.
- A first run of revision 3 with thinking on, seed 42, at a 4,096-token output limit scored 13. Thinking shares that limit, and it
  left the fan-out with no search terms, so thinking runs moved to 16,384 tokens. Those are the scores above.
- Scoring is by alias, the same for every row. On seed 42 two of revision 3's correct answers, questions 17 and 20, also list an
  extra item. A stricter rule would mark them wrong.
- Local check on the tuning set, revision 3, Qwen3.5-9B nvfp4 in Ollama, one run: 17 of 20, median 6 turns, 41k tokens, 16.8 minutes.
- Four of the 20 questions (`adubench_single_001140`, `000385`, `000511`, `001190`) are not in the current XL-DocBench release on
  Hugging Face, which holds 1,345 of the 1,519 questions in the copy used here. Correct answers on the other 16, per seed:

| Configuration | Correct of 16 |
| --- | ---: |
| merged_rag | 3, 3, 3 |
| merged_hybrid | 5, 5, 5 |
| merged_grep_fanout | 3, 3, 2 |
| merged_grep | 5, 7, 5 |
| merged_grep_revision2 | 5, 4, 5 |
| merged_grep_revision3 | 5, 5, 6 |
| merged_grep_revision3_thinking | 9, 6, 5 |
