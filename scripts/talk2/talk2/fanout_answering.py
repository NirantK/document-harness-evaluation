"""Configuration `grep_fanout`: search without an agent. The model writes several search terms once, every term is
searched once, and the model answers from the gathered results in one further call. It cannot search again."""

import json
import re
import time

from talk2.answer_format import ANSWER_INSTRUCTIONS, parse_answer
from talk2.model_client import chat

MAXIMUM_TERMS = 6
FILE_PATH = re.compile(r'<file path="([^"]+)"')
QUERY_PROMPT = """<role>
You write search terms for a question about a document collection. A search program will run each term once as a
case-insensitive regular expression over every file. You will not be able to search again.
</role>

<catalogue>
{catalogue}
</catalogue>

<instructions>
Write up to {maximum} terms. Use distinctive words that a matching line would contain, including words from a title in
the catalogue and the wording of the document's language. Prefer one to three words per term.
Reply with only a JSON list of strings.
</instructions>"""
ANSWER_PROMPT = ("You answer questions about a document collection using only the search results below.\n"
                 + ANSWER_INSTRUCTIONS)


def parse_terms(reply: str) -> list[str]:
    match = re.search(r"\[.*\]", reply, flags=re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            parsed = []
        terms = [str(term).strip() for term in parsed if str(term).strip()] if isinstance(parsed, list) else []
        if terms:
            return terms[:MAXIMUM_TERMS]
    return [line.strip(" -*\"'") for line in reply.splitlines() if line.strip(" -*\"'")][:MAXIMUM_TERMS]


def answer_question(question: dict, tools, model: str, backend: str, think: bool = False) -> dict:
    started = time.time()
    query_turn = chat([{"role": "system", "content": QUERY_PROMPT.format(catalogue=tools.catalogue_text(),
                                                                         maximum=MAXIMUM_TERMS)},
                       {"role": "user", "content": f"Question: {question['question']}"}],
                      model=model, backend=backend, think=think)
    terms = parse_terms(query_turn.content)
    results = [(term, tools.search(term)) for term in terms]
    passages = "\n\n".join(f'<search term="{term}">\n{result}\n</search>' for term, result in results)
    answer_turn = chat([{"role": "system", "content": ANSWER_PROMPT},
                        {"role": "user", "content": f"Search results:\n\n{passages}\n\nQuestion: {question['question']}"}],
                       model=model, backend=backend, think=think)
    return {
        "response": parse_answer(answer_turn.content),
        "raw_reply": answer_turn.content,
        "search_terms": terms,
        "search_results": [{"term": term, "result": result} for term, result in results],
        "files_touched": sorted({path for _, result in results for path in FILE_PATH.findall(result)}),
        "prompt_tokens": query_turn.prompt_tokens + answer_turn.prompt_tokens,
        "completion_tokens": query_turn.completion_tokens + answer_turn.completion_tokens,
        "model_calls": 2, "tool_calls": len(terms), "provider": answer_turn.provider,
        "wall_seconds": round(time.time() - started, 1),
    }
