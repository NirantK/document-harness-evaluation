## Post-freeze changes

None. The agent was not changed after its first run on these questions.

## Observations (first repeat, 2 Nov)

- **`markdown_grep_structured` does not beat plain RAG here.** `rag` 3/10, `markdown_grep` 1/10, `markdown_grep_structured`
  3/10. On the 20 tuning questions the same agent scores 15, against 13 and 14 for `markdown_grep` and 10 for `rag`.
  Most of that tuning gain did not carry over to fresh questions.
- **`markdown_grep` lost #1 and #2 to a tool call Ollama could not parse.** It has no handling for that error. The
  structured agent handles it, and it answered #2 correctly.
- **The structured agent used its whole budget of 20 calls on 5 of 10 questions** (#4, #6, #7, #8, #10). Its median was
  129 s per question against 10 s for `rag`. On #4 it answered `not_found`, where `markdown_grep` found the gold in 10 calls.
- **The gold terms occur in the Markdown of all ten documents.** Conversion does not explain the misses.
- **`rag` on #6 is a lenient hit.** The answer mentions evapotranspiration inside a broader statement, and the alias
  accepts that. A stricter reading would give `rag` 2/10.
- The set is ten questions with one complete repeat per configuration. A difference of one or two questions is noise.
