"""Configuration `markdown_grep_structured`: the grep agent rebuilt around the failures in the `markdown_grep` traces.

The loop adds mechanical checks the model cannot skip: it refuses repeated calls, shows how many calls remain, sends back
an early refusal, checks every quoted line against the cited file, and asks for one review before accepting an answer.
"""

import dataclasses
import json
import re
import time
from dataclasses import dataclass

from talk2.answer_format import parse_answer
from talk2.harness_agent import EMPTY_REPLY_PROMPTS
from talk2.model_client import MalformedToolCall, chat
from talk2.structured_tools import TOOL_DEFINITIONS, StructuredCorpusTools

TOOL_CALL_BUDGET = 20
MINIMUM_SEARCHES_BEFORE_REFUSAL = 4
MALFORMED_CALL_ALLOWANCE = 3

SYSTEM_PROMPT = """<role>
You answer questions about a document collection. You cannot see the files except through the tools. Every fact in your
answer must come from a line you have read with a tool.
</role>

<catalogue>
path | title | language | pages
{catalogue}
</catalogue>

<file_format>
Each file is Markdown converted from selected pages of a document. A <!-- page N --> line marks the start of a page.
Files hold only some pages of the original, so a page number printed in a table of contents is not a line number and
is not a page marker. Use search or outline to find line numbers.
</file_format>

<procedure>
1. Choose files from the catalogue by title. When the question names a document, work inside that file only.
2. Search short terms of one to three words, written as the document would write them. A document in another language
   needs search terms in that language; translate names and technical terms before searching.
3. When the question names a section, call outline, then read that section.
4. Read around every useful hit. A matching line is rarely the whole answer.
5. When the question has several parts or names several sections or documents, read each one before answering.
6. When the question asks which documents match, search two or three phrasings with mode="count", then check every
   candidate file. Stopping at the first hit misses the others.
7. When the question asks whether something exists, a related but different thing does not count. Search several
   wordings, read the likely clauses, then answer "No, ..." if it is absent.
8. Stop calling tools once you can support the answer, then reply with the final JSON.
</procedure>

<final_answer>
Reply with only this JSON object:
{{"answer": "<short answer in the document's exact wording>",
  "files": ["<path from the catalogue>"],
  "found": "yes" | "no" | "not_found",
  "evidence": [{{"file": "<path>", "line": <number>, "quote": "<words copied exactly from that line>"}}]}}
- "files" lists every file that supports the answer. A question asking which documents match lists every match.
- "found" is "yes" when you found the answer or the thing asked about exists, "no" when the documents show it does not
  exist, and "not_found" when you searched thoroughly and the documents do not contain the answer.
- "evidence" holds one short quote for each part of the answer.
</final_answer>"""

CONCISE_PROMPT = """<role>
You answer questions about a document collection in as few turns as you can. You cannot see the files except through
the tools. Every fact in your answer must come from a line you have read with a tool.
</role>

<catalogue>
path | title | language | pages
{catalogue}
</catalogue>

<collection>
Files come from these sources, shown by the first part of the path:
- cuad/: commercial contracts filed as US SEC exhibits. Clauses are numbered. Governing law reads "governed by" or "laws
  of". A restriction may be worded "exclusive", "shall not" or "compete" without using the word in the question.
- vidore-*/: European and French reports on energy, finance, employment and industry. French reports need French terms.
- xl-docbench/: long public reports, standards and books. A question often names the document or a section by its title.
Each file is Markdown converted from selected pages. A <!-- page N --> line marks the start of a page. A page number
printed in a table of contents is not a line number and is not a page marker.
</collection>

<procedure>
A turn is one reply. Every turn is expensive, and you may send several tool calls in one reply. Aim for three turns.
Turn 1: send every search you can already foresee, together, three to six calls. When the question names a document,
pass that file in each search. When it names no document, search all files. Write terms of one to three words as the
document would write them, with | between alternatives. When it asks which documents match, use mode="count".
Turn 2: read the pages that hold the hits, together, with read and page=N. Read every candidate file in this one turn.
Turn 3: call final_answer. Search again only if a part of the question has no supporting line.
When the question asks whether something exists, a related but different thing does not count: answer "No, ..." once
the likely clauses show that it is absent.
</procedure>

<final_answer>
Call final_answer with:
- answer: a short answer in the document's exact wording.
- files: every catalogue path that supports the answer. A question asking which documents match lists every match.
- found: "yes" when you found the answer or the thing exists, "no" when the documents show it does not exist,
  "not_found" when the documents do not contain the answer.
- evidence: one short quote for each part of the answer, each with its file and line, copied exactly from a line you read.
</final_answer>"""
CATALOGUE_OMITTED = "<catalogue>\nShown in your first turn. Use the paths in the results above.\n</catalogue>"

REFUSAL_PROMPT = ("You have run only {searches} searches. Do not give up yet. Search different wording: a single "
                  "distinctive word, a synonym, or the terms in the document's language. Then give the final JSON.")
MISSING_QUOTE_PROMPT = ("These quotes do not occur in the cited files:\n{quotes}\nRead the location again and reply with "
                        "the final JSON, copying each quote exactly from a line you read.")
MALFORMED_CALL_PROMPT = ("Your last tool call could not be parsed. Call one tool again with short plain arguments: no "
                         "angle brackets and no markup in any value. The tools are search, outline and read.")
UNPARSED_PROMPT = "Your reply was not the JSON object. Reply with only the final JSON object."
REVIEW_PROMPT = """<review>
Check your answer before it is accepted:
- Does each part of the question have a quote from a line you read?
- If the question names several documents or sections, did you read each one?
- If the question asks which documents match, did you check every candidate?
If something is missing, call the tools now. Otherwise repeat the final JSON, corrected if needed.
</review>"""


@dataclass(frozen=True)
class AgentRevision:
    """Revision 1 is the agent as frozen under `talk2-structured-frozen`. Revision 2 answers the held-out traces, in which
    the model repeated refused calls until the budget ran out and tried to call a final_answer tool that did not exist."""

    final_answer_tool: bool = False
    stop_after_refused_turns: int = 0
    refused_calls_are_free: bool = False
    think: bool = False
    # Revision 3 fields: the fewest turns and tokens at the same score.
    prompt: str = SYSTEM_PROMPT
    review_step: bool = True
    turn_limit: int = 0
    catalogue_in_first_turn_only: bool = False


REVISION_1 = AgentRevision()
REVISION_2 = AgentRevision(final_answer_tool=True, stop_after_refused_turns=2, refused_calls_are_free=True)
REVISION_2_THINKING = AgentRevision(final_answer_tool=True, stop_after_refused_turns=2, refused_calls_are_free=True,
                                    think=True)
REVISION_3 = AgentRevision(final_answer_tool=True, stop_after_refused_turns=2, refused_calls_are_free=True,
                           prompt=CONCISE_PROMPT, review_step=False, turn_limit=6, catalogue_in_first_turn_only=True)
REVISION_3_THINKING = dataclasses.replace(REVISION_3, think=True)
FINAL_ANSWER = {"type": "function", "function": {
    "name": "final_answer",
    "description": "Submit the answer once the lines you have read support it. This ends the search.",
    "parameters": {"type": "object", "properties": {
        "answer": {"type": "string", "description": "Short answer in the document's exact wording."},
        "files": {"type": "array", "items": {"type": "string"}, "description": "Catalogue paths supporting the answer."},
        "found": {"type": "string", "enum": ["yes", "no", "not_found"]},
        "evidence": {"type": "array", "items": {"type": "object", "properties": {
            "file": {"type": "string"}, "line": {"type": "integer"}, "quote": {"type": "string"}}}},
    }, "required": ["answer", "files", "found"]}}}
REFUSED_PREFIXES = ("You made this exact call", "Lines ")
STOP_SEARCHING_PROMPT = ("Stop searching. Give the final JSON now with the best answer the lines you have read support, "
                         "even if you are unsure. Use not_found only if you read nothing relevant.")


def final_answer_text(arguments: dict) -> str:
    """The final_answer arguments as the JSON reply; the model sometimes writes list arguments as JSON strings."""
    normalised = dict(arguments)
    for key in ("files", "evidence"):
        if isinstance(normalised.get(key), str):
            try:
                normalised[key] = json.loads(normalised[key])
            except json.JSONDecodeError:
                normalised[key] = [normalised[key]] if key == "files" else []
    return json.dumps(normalised, ensure_ascii=False)


def call_key(name: str, arguments: dict) -> str:
    return name + json.dumps(arguments, sort_keys=True, ensure_ascii=False)


class CallLedger:
    """Refuses a call already made and a read of lines already shown, which is how the earlier agent looped."""

    def __init__(self):
        self.calls: dict[str, int] = {}
        self.read_lines: dict[str, set[int]] = {}

    def earlier_step(self, name: str, arguments: dict, step: int) -> int | None:
        key = call_key(name, arguments)
        if key in self.calls:
            return self.calls[key]
        self.calls[key] = step
        return None

    def already_read(self, path: str, line_range: tuple[int, int]) -> bool:
        return set(range(line_range[0], line_range[1] + 1)) <= self.read_lines.get(path, set())

    def record_read(self, path: str, line_range: tuple[int, int]) -> None:
        self.read_lines.setdefault(path, set()).update(range(line_range[0], line_range[1] + 1))


def execute_call(call: dict, tools: StructuredCorpusTools, ledger: CallLedger, step: int) -> str:
    name, arguments = call["name"], call["arguments"] if isinstance(call["arguments"], dict) else {}
    earlier = ledger.earlier_step(name, arguments, step)
    if earlier is not None:
        return f"You made this exact call at step {earlier}; the result is unchanged. Change the terms or the lines."
    if name == "search":
        return tools.search(pattern=arguments["pattern"], file=arguments.get("file") or arguments.get("path") or "",
                            mode=arguments.get("mode") or "lines")
    if name == "outline":
        return tools.outline(arguments.get("path", ""))
    if name == "read":
        selected = {k: v for k, v in arguments.items() if k in ("start_line", "line_count", "page")}
        path = arguments.get("path", "")
        requested = tools.line_range(path, selected.get("start_line"), selected.get("line_count"), selected.get("page"))
        if requested and ledger.already_read(path, requested):
            return (f"Lines {requested[0]}-{requested[1]} of {path} are already above. Read other lines, "
                    "search again, or answer.")
        text, shown = tools.read(path, **selected)
        if shown:
            ledger.record_read(path, shown)
        return text
    return f"Unknown tool {name}. The tools are search, outline and read."


def missing_quotes(response: dict, tools: StructuredCorpusTools) -> list[str]:
    missing = []
    for item in response.get("evidence") or []:
        if isinstance(item, dict) and item.get("quote") and not tools.quote_is_present(str(item.get("file", "")),
                                                                                       str(item["quote"])):
            missing.append(f"- {item.get('file')}: {str(item['quote'])[:160]}")
    return missing


def answer_question(question: dict, tools: StructuredCorpusTools, model: str, backend: str,
                    revision: AgentRevision = REVISION_1) -> dict:
    started = time.time()
    log: list[dict] = []
    prompt_tokens = completion_tokens = model_calls = parse_failures = 0
    files_touched: set[str] = set()
    ledger = CallLedger()
    system = revision.prompt.format(catalogue=tools.catalogue_text())
    messages = [{"role": "system", "content": system}, {"role": "user", "content": f"Question: {question['question']}"}]
    tool_calls_made = searches = empty_replies = malformed_calls = refused_turns = 0
    tool_definitions = TOOL_DEFINITIONS + ([FINAL_ANSWER] if revision.final_answer_tool else [])
    stop_searching = False
    checks_used: set[str] = set()
    accepted: dict | None = None
    final_text = ""

    while True:
        if revision.turn_limit and model_calls >= revision.turn_limit - 1:
            stop_searching = True
        if revision.catalogue_in_first_turn_only and model_calls == 1:
            messages[0]["content"] = re.sub(r"<catalogue>.*?</catalogue>", CATALOGUE_OMITTED, system, flags=re.DOTALL)
        budget_left = tool_calls_made < TOOL_CALL_BUDGET and not stop_searching
        if not budget_left:
            messages.append({"role": "user", "content": STOP_SEARCHING_PROMPT if revision.stop_after_refused_turns
                             else "The tool budget is used up. Give the final JSON now."})
        model_calls += 1
        try:
            turn = chat(messages, tools=tool_definitions if budget_left else None, model=model, backend=backend,
                        think=revision.think)
        except MalformedToolCall as error:
            malformed_calls += 1
            parse_failures += 1
            log.append({"step": "malformed_tool_call", "error": str(error)[:200]})
            if malformed_calls > MALFORMED_CALL_ALLOWANCE:
                tool_calls_made = TOOL_CALL_BUDGET
            else:
                messages.append({"role": "user", "content": MALFORMED_CALL_PROMPT})
            continue
        prompt_tokens += turn.prompt_tokens
        completion_tokens += turn.completion_tokens

        submitted = next((c for c in turn.tool_calls if c["name"] == "final_answer"), None)
        if budget_left and revision.final_answer_tool and submitted:
            turn.content = final_answer_text(submitted["arguments"] if isinstance(submitted["arguments"], dict) else {})
            turn.tool_calls = []
        if budget_left and turn.tool_calls:
            empty_replies = 0
            messages.append({"role": "assistant", "content": turn.content, "tool_calls": turn.tool_calls,
                             "thinking": turn.thinking})
            refused_in_turn = False
            for call in turn.tool_calls:
                arguments = call["arguments"] if isinstance(call["arguments"], dict) else {}
                searches += call["name"] == "search"
                tool_calls_made += 1
                try:
                    output = execute_call(call, tools, ledger, tool_calls_made)
                except (KeyError, TypeError, ValueError) as error:
                    parse_failures += 1
                    output = f"Bad arguments: {error}"
                if call["name"] == "read" and arguments.get("path"):
                    files_touched.add(arguments["path"])
                if output.startswith(REFUSED_PREFIXES):
                    refused_in_turn = True
                    if revision.refused_calls_are_free:
                        tool_calls_made -= 1
                log.append({"tool": call["name"], "arguments": arguments, "output": output[:2500]})
                output += f"\n<progress calls_used=\"{tool_calls_made}\" calls_limit=\"{TOOL_CALL_BUDGET}\"/>"
                messages.append({"role": "tool", "content": output, "tool_call_id": call["id"], "name": call["name"]})
            refused_turns = refused_turns + 1 if refused_in_turn else 0
            if revision.stop_after_refused_turns and refused_turns >= revision.stop_after_refused_turns:
                stop_searching = True
                log.append({"step": "stop_searching", "refused_turns": refused_turns})
            continue

        if not turn.content.strip() and empty_replies < len(EMPTY_REPLY_PROMPTS):
            messages += [{"role": "assistant", "content": ""},
                         {"role": "user", "content": EMPTY_REPLY_PROMPTS[empty_replies]}]
            empty_replies += 1
            parse_failures += 1
            log.append({"step": "empty_reply", "reprompt": empty_replies})
            continue

        final_text = turn.content
        response = parse_answer(final_text)
        if not response.get("parse_error"):
            accepted = response
        check_prompt = ""
        if tool_calls_made < TOOL_CALL_BUDGET and not stop_searching:
            if response.get("parse_error") and "format" not in checks_used:
                checks_used.add("format")
                check_prompt = UNPARSED_PROMPT
            elif response.get("found") != "yes" and searches < MINIMUM_SEARCHES_BEFORE_REFUSAL \
                    and "refusal" not in checks_used:
                checks_used.add("refusal")
                check_prompt = REFUSAL_PROMPT.format(searches=searches)
            elif missing_quotes(response, tools) and "quotes" not in checks_used:
                checks_used.add("quotes")
                check_prompt = MISSING_QUOTE_PROMPT.format(quotes="\n".join(missing_quotes(response, tools)))
            elif revision.review_step and "review" not in checks_used:
                checks_used.add("review")
                check_prompt = REVIEW_PROMPT
        if not check_prompt:
            break
        log.append({"step": "check", "prompt": check_prompt[:200], "reply": final_text[:800]})
        messages += [{"role": "assistant", "content": final_text}, {"role": "user", "content": check_prompt}]

    response = parse_answer(final_text)
    if response.get("parse_error") and accepted:
        response = accepted
    return {
        "response": response, "raw_reply": final_text, "trace": log,
        "files_touched": sorted(files_touched | set(response.get("files", []))),
        "tool_calls": tool_calls_made, "tool_call_parse_failures": parse_failures,
        "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "model_calls": model_calls,
        "provider": backend, "wall_seconds": round(time.time() - started, 1),
    }
