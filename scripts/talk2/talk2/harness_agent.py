"""Configurations `markdown_grep` (DSH steps 1+2) and `markdown_grep_memory` (DSH steps 1+2+3)."""

import json
import time

from talk2.answer_format import ANSWER_INSTRUCTIONS, parse_answer
from talk2.harness_tools import LIST_FILES, READ, RECORD_NOTE, CorpusTools
from talk2.model_client import chat

TOOL_CALL_BUDGET = 20
RECENT_TOOL_OUTPUTS_KEPT = 3
# Ollama returns neither content nor a tool call when it cannot parse a tool call, which is not a final answer.
EMPTY_REPLY_PROMPTS = (
    "Your last reply was empty or could not be parsed. Call one tool again with plain arguments, without quotation "
    "marks, or give the final JSON.",
    "Your last reply was again empty or could not be parsed. Call list_files with no arguments, or give the final JSON.",
)

SYSTEM_PROMPT = """You answer questions about a document collection stored as text files. You cannot see the files
directly; use the tools. search is exhaustive over every file, so it can show that something is absent. Search for
distinctive literal strings (names, numbers, identifiers) and read around the hits before answering. When you have
the answer, stop calling tools and reply with the final JSON.
""" + ANSWER_INSTRUCTIONS

MEMORY_PROMPT = """
Use record_note to save each fact you establish (with its file and line) before moving on. Older tool outputs are
removed from view after a few steps, but your notes stay. Notes from earlier questions in this session are listed
below; check them first."""

TRANSLATION_PROMPT = """The collection contains documents in these languages: {languages}.
List the search terms you would grep for to answer the question below, written in each of those languages as they
would appear in the documents (names, technical terms, units, likely phrasing). Reply with only a JSON object
{{"<language>": ["term", ...], ...}}.

Question: {question}"""


class SessionMemory:
    """Notes and question-to-file records that persist across questions within one run of the ten questions."""

    def __init__(self):
        self.notes: list[dict] = []
        self.answered_from: list[dict] = []

    def render(self) -> str:
        lines = [f"- {n['fact']} (file: {n['file']}, line: {n.get('line', '?')})" for n in self.notes[-40:]]
        lines += [f"- Question {a['order']} ('{a['question'][:80]}') was answered from: {', '.join(a['files'])}"
                  for a in self.answered_from]
        return "\n".join(lines) if lines else "(no notes yet)"


def elide_old_tool_outputs(messages: list[dict]) -> None:
    tool_indexes = [i for i, m in enumerate(messages) if m["role"] == "tool"]
    for index in tool_indexes[:-RECENT_TOOL_OUTPUTS_KEPT]:
        if not messages[index]["content"].startswith("[elided"):
            messages[index]["content"] = "[elided older tool output; see your notes]"


def answer_question(question: dict, tools: CorpusTools, model: str, backend: str,
                    memory: SessionMemory | None = None) -> dict:
    started = time.time()
    use_memory = memory is not None
    log: list[dict] = []
    prompt_tokens = completion_tokens = model_calls = parse_failures = 0
    files_touched: set[str] = set()

    user_content = f"Question: {question['question']}"
    if use_memory:
        languages = sorted({r["language"] for r in tools.catalogue if r["language"]})
        turn = chat([{"role": "user", "content": TRANSLATION_PROMPT.format(languages=", ".join(languages),
                                                                          question=question["question"])}],
                    model=model, backend=backend)
        model_calls += 1
        prompt_tokens += turn.prompt_tokens
        completion_tokens += turn.completion_tokens
        log.append({"step": "translation", "reply": turn.content})
        user_content += f"\n\nSuggested search terms by document language:\n{turn.content.strip()}"

    def system_message() -> dict:
        content = tools.system_prompt or SYSTEM_PROMPT
        if use_memory:
            content += MEMORY_PROMPT + "\n\nNotes:\n" + memory.render()
        return {"role": "system", "content": content}

    tool_definitions = [tools.search_definition, READ, LIST_FILES] + ([RECORD_NOTE] if use_memory else [])
    messages = [system_message(), {"role": "user", "content": user_content}]
    tool_calls_made = 0
    empty_replies = 0
    final_text = ""
    while True:
        budget_left = tool_calls_made < TOOL_CALL_BUDGET
        if not budget_left:
            messages.append({"role": "user", "content": "Tool budget exhausted. Give your final JSON answer now."})
        if use_memory:
            messages[0] = system_message()
            elide_old_tool_outputs(messages)
        turn = chat(messages, tools=tool_definitions if budget_left else None, model=model, backend=backend)
        model_calls += 1
        prompt_tokens += turn.prompt_tokens
        completion_tokens += turn.completion_tokens
        if budget_left and not turn.tool_calls and not turn.content.strip() and empty_replies < len(EMPTY_REPLY_PROMPTS):
            messages += [{"role": "assistant", "content": ""}, {"role": "user", "content": EMPTY_REPLY_PROMPTS[empty_replies]}]
            empty_replies += 1
            parse_failures += 1
            log.append({"step": "empty_reply", "reprompt": empty_replies})
            continue
        if not turn.tool_calls or not budget_left:
            final_text = turn.content
            break
        empty_replies = 0
        messages.append({"role": "assistant", "content": turn.content, "tool_calls": turn.tool_calls})
        for call in turn.tool_calls:
            tool_calls_made += 1
            arguments = call["arguments"] if isinstance(call["arguments"], dict) else {}
            if "_unparsed" in arguments or not isinstance(call["arguments"], dict):
                parse_failures += 1
            try:
                if call["name"] == "search":
                    output = tools.search(**{k: v for k, v in arguments.items() if k in tools.search_parameters})
                elif call["name"] == "read":
                    output = tools.read(**{k: v for k, v in arguments.items() if k in ("path", "start_line", "line_count")})
                    files_touched.add(arguments.get("path", ""))
                elif call["name"] == "list_files":
                    output = tools.list_files(arguments.get("glob", "**"))
                elif call["name"] == "record_note" and use_memory:
                    memory.notes.append({k: arguments.get(k) for k in ("fact", "file", "line")})
                    output = "noted"
                else:
                    output = f"unknown tool {call['name']}"
            except TypeError as error:
                parse_failures += 1
                output = f"bad arguments: {error}"
            log.append({"tool": call["name"], "arguments": arguments, "output": output[:2500]})
            messages.append({"role": "tool", "content": output, "tool_call_id": call["id"], "name": call["name"]})

    response = parse_answer(final_text)
    if use_memory and response.get("files"):
        memory.answered_from.append({"order": question["order"], "question": question["question"],
                                     "files": response["files"]})
    return {
        "response": response, "raw_reply": final_text, "trace": log,
        "files_touched": sorted(files_touched | set(response.get("files", []))),
        "tool_calls": tool_calls_made, "tool_call_parse_failures": parse_failures,
        "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "model_calls": model_calls,
        "provider": backend, "wall_seconds": round(time.time() - started, 1),
    }
