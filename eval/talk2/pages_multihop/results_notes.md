## Post-freeze changes

Every change after the `talk2-multihop-frozen` tag:
1. **Run stability (2 Nov).** The Mac restarted during the first pass on 1 Nov. A generation in `gemma4:12b`'s
   `markdown_grep_memory` hit the 30-minute request timeout, and that partial log is discarded. From repeat 1 of that
   configuration onward:
   - Each model call stops at 4,096 output tokens. The most any completed question has used across all its calls is
     2,511, so completed logs are unaffected.
   - The answer model is unloaded after each question, because the MLX runner's memory grew from about 8 GB to 17 GB
     while it stayed loaded. This does not change the prompts or settings.
   - Calls pause when free memory is under 15% or free disk under 8 GiB, and retry when Ollama is down, so an outage
     is not scored as a wrong answer.
2. **Scorer corrections (2 Nov).** Both are false negatives found by reading the wrong answers, and the report re-scores
   every log:
   - An unseparated number such as "2141" was read as "214" and "1", so a correct #2 was marked wrong (qwen3.5:9b,
     `markdown_grep` and `markdown_hybrid`).
   - #7 also accepts "does not contain a non-compete", under the 1 Nov decision, not only an answer starting "No".
3. **Gemma build and empty replies (2 Nov).** The first pass with `gemma4:12b` (MLX, nvfp4) is invalid for grep: 26 of 46
   grep replies were empty because Ollama dropped tool calls it could not parse, and the harness read each as a final
   answer. Those logs are in `runs_superseded/`. Two changes:
   - Gemma now runs as `gemma4:12b-it-q4_K_M` (GGUF, llama.cpp runner) in every configuration. It is the same model
     size class, and its replies repeat exactly at temperature 0 with a fixed seed. `qwen3.5:9b` stays on MLX, so the
     two columns differ in runner as well as in model.
   - The grep agent re-prompts up to twice after an empty reply and counts it in `tool_call_parse_failures`. No
     `qwen3.5:9b` log has an empty reply, so its completed logs are unaffected. On Gemma the re-prompt recovered all five
     first turns that were still empty on the GGUF build.

4. **Structured grep agent (2 Nov, tag `talk2-structured-frozen`).** `markdown_grep_structured` is a new configuration for
   `qwen3.5:9b`. It was written after reading the failures of `markdown_grep` on these same 20 questions, so its score
   here is a tuning score. `../pages_holdout/` holds the test of whether it generalises. No prompt or tool names a
   question, a document or an answer. Each change answers a failure class seen in the traces:
   - Search ignores case, because title and heading searches returned 0 matches on case alone.
   - The catalogue of paths and titles is in the system prompt, because the old agent spent its first calls guessing
     paths from titles.
   - An `outline` tool and page numbers on every hit, and the prompt says a contents-page number is not a line number,
     because one question re-read the same lines seven times after treating "page 140" as line 140.
   - The loop refuses repeated calls and repeated reads, and each tool result shows the calls used.
   - A refusal after fewer than four searches is sent back, as is a quote that does not occur in the cited file.
   - One review turn before an answer is accepted.
   - A tool call Ollama cannot parse now returns a correction to the model instead of 20 identical retries. The old
     agent lost `text_grep` repeat 2 #14 to that error, and #18 to a non-numeric line argument.
   Sampling and thinking are unchanged from the other configurations (temperature 0, seed 42, thinking off, the
   Modelfile's `presence_penalty` 1.5).

5. **The tools cut long lines with no notice (found 2 Nov 2026, after the structured runs).** Question 6 scored 5 of 6 with the structured agent. The missed NICE contract has clause "17.11. LAW AND JURISDICTION ... laws of the state of Israel" at character 2,197 of a 4,248-character line, because the converter merged clauses 17.9 to 17.11 into one line. `search` displayed 400 characters of a line and `read` 600, in both the old and the structured agent, and neither said so. The agent found and opened the right file, then could not see the clause. 23 of the 68 Markdown files hold a line over 1,000 characters. Revision 2 shows excerpts around each match and whole lines on `read` (tag `talk2-revision2-long-lines-frozen`). A smoke test on question 6 then named the NICE contract in 12 calls, with the clause quoted. Revision 1 keeps the cut display, so its logged scores stay reproducible. All grep scores before revision 2 carry this limit.
