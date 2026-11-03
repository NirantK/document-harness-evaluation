"""Automatic scoring of one answer against the match rule in questions.jsonl."""

import json
import re
import unicodedata
from pathlib import Path

from talk2.corpus_assembly import QUESTIONS
NUMBER = re.compile(r"\d{1,3}(?:[   ,.]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?")
STATED_ABSENCE = re.compile(r"(?:does not|doesn't|do not) (?:contain|include|have)\b[^.]{0,40}non-?compet|\bno non-?compet",
                            flags=re.IGNORECASE)


def load_questions() -> list[dict]:
    return [json.loads(line) for line in QUESTIONS.open()]


def normalise(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).lower()


def parse_numbers(text: str) -> list[float]:
    values = []
    for token in NUMBER.findall(unicodedata.normalize("NFKC", text)):
        token = token.replace(" ", " ").replace(" ", " ")
        if re.fullmatch(r"\d{1,3}([ ,]\d{3})+(\.\d+)?", token):
            token = token.replace(" ", "").replace(",", "")
        elif re.fullmatch(r"\d{1,3}( \d{3})+(,\d+)?", token):
            token = token.replace(" ", "").replace(",", ".")
        else:
            token = token.replace(" ", "").replace(",", ".")
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values


def file_stem(name: str) -> str:
    stem = Path(name.strip()).name
    for suffix in (".md", ".txt", ".pdf", ".PDF"):
        stem = stem.removesuffix(suffix)
    return stem


def score(question: dict, response: dict) -> dict:
    """Return {'correct': bool, 'detail': str} for a response {'answer', 'files', 'found'}."""
    match = question["match"]
    answer = str(response.get("answer", ""))
    rule = match["rule"]
    if rule == "exact_string":
        hit = normalise(match["value"]) in normalise(answer)
        rejected = [r for r in match.get("reject", []) if normalise(r) in normalise(answer)]
        return {"correct": hit and not rejected, "detail": f"hit={hit} rejected={rejected}"}
    if rule == "numeric":
        numbers = parse_numbers(answer)
        low = match.get("low", match.get("value", 0) - match.get("tolerance", 0))
        high = match.get("high", match.get("value", 0) + match.get("tolerance", 0))
        within = any(low <= n <= high for n in numbers)
        rejected = [r for r in match.get("reject", []) if any(abs(n - r) < 0.01 for n in numbers)]
        return {"correct": within and not rejected, "detail": f"numbers={numbers[:6]} range=[{low},{high}] rejected={rejected}"}
    if rule == "alias_set":
        hit = [a for a in match["aliases"] if a.lower() in answer.lower()]
        return {"correct": bool(hit), "detail": f"aliases={hit}"}
    if rule == "all_terms":
        lowered = answer.lower()
        missing = [terms[0] for terms in match["terms"] if not any(term in lowered for term in terms)]
        return {"correct": not missing, "detail": f"missing={missing}"}
    if rule == "range_both_bounds":
        lowered = answer.lower()
        bounds = {bound: any(re.search(rf"\b{form}\b", lowered) for form in forms)
                  for bound, forms in match["word_forms"].items()}
        return {"correct": all(bounds.values()), "detail": f"bounds={bounds}"}
    if rule == "file_set":
        reported = {file_stem(f) for f in response.get("files", [])}
        gold = set(match["files"])
        precision = len(reported & gold) / len(reported) if reported else 0.0
        recall = len(reported & gold) / len(gold)
        return {"correct": reported == gold, "detail": f"precision={precision:.2f} recall={recall:.2f}"}
    if rule == "found_no":
        # Decided 1 Nov: an answer that states the absence counts, whatever its found field says, provided it
        # cites the document that was searched.
        states_absence = (response.get("found") == "no" or bool(re.match(r"\s*no\b", answer, flags=re.IGNORECASE))
                          or bool(STATED_ABSENCE.search(answer)))
        cited = {file_stem(f) for f in response.get("files", [])} & {file_stem(d) for d in question["evidence_docs"]}
        return {"correct": states_absence and bool(cited),
                "detail": f"found={response.get('found')} states_absence={states_absence} cited={bool(cited)}"}
    raise ValueError(f"unknown rule {rule}")


def self_test() -> list[str]:
    """Gold-shaped answers must score 1 and known wrong answers 0. Returns failure descriptions."""
    questions = {q["id"]: q for q in load_questions()}
    cuad6 = questions["cuad:israel-governing-law"]["match"]["files"]
    ultragenyx = questions["cuad:ultragenyx-cremer-noncompete"]["evidence_docs"][0]
    adaptimmune = questions["cuad:adaptimmune-mdanderson-19644000"]["match"]["files"]
    cases = [
        ("adubench_cross_000108", {"answer": "E/INCB/2022/1/Supp.1"}, True),
        ("adubench_cross_000108", {"answer": "E/INCB/2022/1"}, False),
        ("energy:596", {"answer": "2 141 MWh/an"}, True),
        ("energy:596", {"answer": "2,141 MWh per year"}, True),
        ("energy:596", {"answer": "2 814 MWh/an"}, False),
        ("hr:211", {"answer": "About 50 EUR per hour"}, True),
        ("hr:211", {"answer": "About 49 EUR per hour"}, False),
        ("energy:442", {"answer": "The Netherlands (Pays-Bas)"}, True),
        ("energy:442", {"answer": "Norway"}, False),
        ("industrial:85+industrial:150", {"answer": "four to five inches"}, True),
        ("industrial:85+industrial:150", {"answer": "4-5 inches"}, True),
        ("industrial:85+industrial:150", {"answer": "6 inches"}, False),
        ("cuad:israel-governing-law", {"files": [f + ".md" for f in cuad6]}, True),
        ("cuad:israel-governing-law", {"files": cuad6[:5]}, False),
        ("cuad:ultragenyx-cremer-noncompete", {"found": "no", "files": [ultragenyx + ".md"]}, True),
        ("cuad:ultragenyx-cremer-noncompete", {"answer": "No, it has none.", "found": "yes",
                                                "files": [ultragenyx]}, True),
        ("cuad:ultragenyx-cremer-noncompete", {"answer": "The agreement does not contain a non-compete clause.",
                                                "found": "yes", "files": [ultragenyx]}, True),
        ("cuad:ultragenyx-cremer-noncompete", {"answer": "The documents do not contain that information.",
                                                "found": "yes", "files": [ultragenyx]}, False),
        ("cuad:ultragenyx-cremer-noncompete", {"answer": "Yes, clause 3.", "found": "yes",
                                                "files": [ultragenyx]}, False),
        ("cuad:ultragenyx-cremer-noncompete", {"answer": "not_found", "found": "not_found", "files": [ultragenyx]},
         False),
        ("cuad:ultragenyx-cremer-noncompete", {"answer": "No.", "found": "no", "files": []}, False),
        ("energy:596", {"answer": "2141"}, True),
        ("energy:596", {"answer": "2 141 MWh/an"}, True),
        ("energy:596", {"answer": "2,141"}, True),
        ("finance_fr:354", {"answer": "97,50 %"}, True),
        ("finance_fr:354", {"answer": "48,60 %"}, False),
        ("cuad:adaptimmune-mdanderson-19644000", {"files": ["cuad/" + adaptimmune[0] + ".md"]}, True),
        ("cuad:adaptimmune-mdanderson-19644000", {"files": []}, False),
        ("adubench_cross_000018", {"answer": "2021"}, True),
        ("adubench_cross_000018", {"answer": "2014"}, False),
    ]
    for question in questions.values():
        if question.get("set") == "multi_hop":
            match = question["match"]
            gold = " and ".join(terms[0] for terms in match["terms"]) if "terms" in match else match["aliases"][0]
            cases += [(question["id"], {"answer": gold}, True), (question["id"], {"answer": "not_found"}, False)]
    failures = []
    for question_id, response, expected in cases:
        result = score(questions[question_id], response)
        if result["correct"] != expected:
            failures.append(f"{question_id} {response} expected={expected} {result['detail']}")
    return failures
