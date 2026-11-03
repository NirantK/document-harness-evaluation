# Results: pages_holdout

10 questions. Model: Qwen3.5-9B, 4-bit (fp4), Darkbloom through OpenRouter (`qwen/qwen3.5-9b`), temperature 0. Each run sends a different seed (41 + run number). Temperature 0 leaves the seed with little effect, so spread between runs most likely comes from the hosted endpoint.

| Iteration | Configuration in the logs | Correct per run | Median turns | Median tokens |
| --- | --- | ---: | ---: | ---: |
| Improving Correctness | `merged_grep_revision2` | 6 | 14.5 | 162.1k |
| Improving Speed | `merged_grep_revision3` | 5 | 13 | 97.7k |
