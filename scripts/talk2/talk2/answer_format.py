"""The answer instructions and JSON schema shared by every configuration, and the parser for final answers."""

import json
import re

ANSWER_INSTRUCTIONS = """Reply with only a JSON object, no other text:
{"answer": "<short answer>", "files": ["<corpus file path>", ...], "found": "yes" | "no" | "not_found"}
- "files" lists the corpus file paths (as shown to you) that support the answer. For a question asking which
  documents match, list every matching file.
- "found" is "yes" when you found the answer or the thing asked about exists, "no" when the documents show it does
  not exist, and "not_found" when the documents you saw do not contain the answer."""


def parse_answer(text: str) -> dict:
    """Extract the last JSON object from the model's reply; fall back to the raw text as the answer."""
    cleaned = re.sub(r"```(?:json)?", "", text)
    for match in reversed(list(re.finditer(r"\{.*\}", cleaned, flags=re.DOTALL))):
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            parsed.setdefault("answer", "")
            parsed.setdefault("files", [])
            parsed.setdefault("found", "not_found")
            parsed["answer"] = str(parsed["answer"])
            return parsed
    return {"answer": text.strip(), "files": [], "found": "not_found", "parse_error": True}
