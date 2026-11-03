# Results: pages_unseen

20 questions. Model: Qwen3.5-9B, 4-bit (fp4), Darkbloom through OpenRouter (`qwen/qwen3.5-9b`), temperature 0. Each run sends a different seed (41 + run number). Temperature 0 leaves the seed with little effect, so spread between runs most likely comes from the hosted endpoint.

| Iteration | Configuration in the logs | Correct per run | Median turns | Median tokens |
| --- | --- | ---: | ---: | ---: |
| RAG, embedding | `merged_rag` | 5, 5, 5 | 1 | 2.6k |
| RAG, embedding, thinking | `merged_rag_thinking` | 8, 7, 5 | 1 | 4.5k |
| RAG, hybrid | `merged_hybrid` | 7, 7, 6 | 1 | 2.7k |
| Grep, single fan-out | `merged_grep_fanout` | 5, 5, 4 | 2 | 6.6k |
| Grep, single fan-out, thinking | `merged_grep_fanout_thinking` | 4, 4, 2 | 2 | 20.5k |
| Grep, agentic search | `merged_grep` | 7, 9, 7 | 20 | 155.3k |
| Improving Correctness | `merged_grep_revision2` | 7, 6, 7 | 13 | 138.2k |
| Improving Speed | `merged_grep_revision3` | 8, 7, 8 | 13 | 98.1k |
| Thinking | `merged_grep_revision3_thinking` | 11, 8, 8 | 11 | 94.5k |

## Notes

- All 20 questions come from XL-DocBench; 12 of the 20 tuning questions do.
- Improving Speed (`merged_grep_revision3`) was frozen under the tag `talk2-revision3-frozen` before its first run here. No run in this table replayed a saved reply.
- A first Thinking run at a 4,096-token output limit scored 13. Thinking shares that limit, and it
  left the fan-out with no search terms, so thinking runs moved to 16,384 tokens. Those are the scores above. Even at that limit, fan-out with thinking ended with no search terms on 5 to 7 of these 20 questions per run, and 2 to 5 of the tuning 20, mostly with reasoning at the limit.
- Scoring is by alias, the same for every row. In the first run two of Improving Speed's correct answers, questions 17 and 20, also list an
  extra item. A stricter rule would mark them wrong.
- Local check on the tuning set, Improving Speed, Qwen3.5-9B nvfp4 in Ollama, one run: 17 of 20, median 6 turns, 41k tokens, 16.8 minutes.
- Four of the 20 questions (`adubench_single_001140`, `000385`, `000511`, `001190`) are not in the current XL-DocBench release on
  Hugging Face, which holds 1,345 of the 1,519 questions in the copy used here. Correct answers on the other 16, per run:

| Configuration | Correct of 16 |
| --- | ---: |
| merged_rag | 3, 3, 3 |
| merged_hybrid | 5, 5, 5 |
| merged_grep_fanout | 3, 3, 2 |
| merged_grep | 5, 7, 5 |
| merged_grep_revision2 | 5, 4, 5 |
| merged_grep_revision3 | 5, 5, 6 |
| merged_grep_revision3_thinking | 9, 6, 5 |
