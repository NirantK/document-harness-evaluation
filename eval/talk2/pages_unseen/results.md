# Results: pages_unseen

20 questions. Model: Qwen3.5-9B, 4-bit (fp4), Darkbloom through OpenRouter (`qwen/qwen3.5-9b`), temperature 0. Each run sends a different seed (41 + run number). Temperature 0 leaves the seed with little effect, so spread between runs most likely comes from the hosted endpoint.

| Iteration | Configuration in the logs | Correct per run | Median turns | Median tokens |
| --- | --- | ---: | ---: | ---: |
| PDF text, RAG, embedding | `rag` | 7, 6, 7 | 1 | 2.4k |
| PDF text, RAG, hybrid | `text_hybrid` | 7, 7, 7 | 1 | 2.6k |
| PDF text, grep, agentic search | `text_grep` | 9, 6, 8 | 21 | 134.1k |
| PDF text, BM25 agent | `text_bm25_loop` | 9, 9, 9 | 16 | 171.0k |
| RAG, embedding | `merged_rag` | 5, 5, 5 | 1 | 2.6k |
| RAG, embedding, thinking | `merged_rag_thinking` | 8, 7, 5 | 1 | 4.5k |
| RAG, hybrid | `merged_hybrid` | 7, 7, 6 | 1 | 2.7k |
| Grep, single fan-out | `merged_grep_fanout` | 5, 5, 4 | 2 | 6.6k |
| Grep, single fan-out, thinking | `merged_grep_fanout_thinking` | 4, 4, 2 | 2 | 20.5k |
| Grep agent | `merged_grep` | 7, 9, 7 | 20 | 155.3k |
| BM25 agent | `merged_bm25_loop` | 9, 8, 10 | 13 | 125.5k |
| BM25 agent + Thinking | `merged_bm25_loop_thinking` | 10, 6, 5 | 21 | 303.9k |
| Hybrid in the loop | `merged_fusion_loop` | 8, 9, 10 | 19.5 | 238.5k |
| Trace Fixes | `merged_grep_revision2` | 7, 6, 7 | 13 | 138.2k |
| Grep Token Reduction | `merged_grep_revision3` | 8, 7, 8 | 13 | 98.1k |
| Grep Token Reduction + Thinking | `merged_grep_revision3_thinking` | 11, 8, 8 | 11 | 94.5k |
| Grep Token Reduction, hint removed | `merged_grep_revision3_bare_empty_result` | 6, 9, 9 | 12 | 98.1k |
| BM25 Token Reduction | `merged_bm25_excerpt_fixed_count` | 7, 9, 8 | 11 | 97.3k |
| BM25 Token Reduction, model-selected count | `merged_bm25_excerpt_selected_count` | 7, 6, 9 | 11.5 | 110.9k |

## Notes

- All 20 questions come from XL-DocBench; 12 of the 20 validation questions do.
- Grep Token Reduction (`merged_grep_revision3`) was frozen under the tag `talk2-revision3-frozen` before its first run here. No run in this table replayed a saved reply.
- A first Thinking run at a 4,096-token output limit scored 13. Thinking shares that limit, and it
  left the fan-out with no search terms, so thinking runs moved to 16,384 tokens. Those are the scores above. Even at that limit, fan-out with thinking ended with no search terms on 5 to 7 of these 20 questions per run, and 3 to 5 of the validation 20, mostly with reasoning at the limit.
- Scoring is by alias, the same for every row. In the first run two of Token Reduction's correct answers, questions 17 and 20, also list an
  extra item. A stricter rule would mark them wrong.
- Local check on the validation set, Token Reduction, Qwen3.5-9B nvfp4 in Ollama, one run: 17 of 20, median 6 turns, 41k tokens, 16.8 minutes.
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

## Empty-result hint

`merged_grep_revision3_bare_empty_result` is Token Reduction with the guidance and term counts removed from empty search results, a
measurement and not a candidate. Three runs each, Qwen3.5-9B, 4-bit (fp4), Darkbloom through OpenRouter. Totals are off → on.

| Source, set | Answers | Empty searches | Same search repeated | Correct |
| --- | ---: | ---: | ---: | ---: |
| ViDoRe, validation | 15 | 63 → 32 | 20 → 1 | 11 → 10 |
| CUAD, validation | 9 | 9 → 6 | 0 → 0 | 9 → 9 |
| XL-DocBench, validation | 36 | 38 → 34 | 3 → 3 | 25 → 21 |
| XL-DocBench, test | 60 | 86 → 83 | 0 → 2 | 24 → 23 |

The hint lowers empty and repeated searches and does not raise the score: validation scores 15, 15, 15 without it and 12, 13, 15 with it, on the shared 531 pages. The hint came from the traces of 5 ViDoRe questions on the earlier 222-page corpus. Three logs from the first validation attempt lost one question each to a
network outage; they were set aside and all three runs repeated.

- BM25 agent (`merged_bm25_loop`), its thinking variant (`merged_bm25_loop_thinking`) and Hybrid in the loop (`merged_fusion_loop`) fork the plain grep agent (`merged_grep`), changing only the search tool: it returns the five best passages from the chunks plain RAG uses, ranked by BM25 alone or by Hybrid, meaning embedding + BM25 with reciprocal rank fusion, top 5. These were added after the test scores of the other rows were known, with no change made for any question, so this comparison is not a clean holdout.
- BM25 Token Reduction (`merged_bm25_excerpt_fixed_count`) puts BM25 search into the Grep Token Reduction agent: each result is a 600-character excerpt around the matching words with its line number, there is no turn limit, and the search stops after 3 refused turns. `merged_bm25_excerpt_selected_count` lets the model choose the passage count, 1 to 10. Both were tuned on the validation questions and added after the test scores of the other rows were known, so this comparison is not a clean holdout. `merged_bm25_token_reduction` and `merged_bm25_selected_count` are their first versions, run on validation only, before that tuning.
