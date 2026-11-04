"""Configurations with a ranked search tool in place of grep.

`merged_bm25_loop`, `merged_bm25_loop_thinking`, `text_bm25_loop` and `merged_fusion_loop` change only the search tool of the
grep agent. Display names: BM25 agent, BM25 agent + Thinking, BM25 agent on PDF text, Hybrid agent.

`merged_bm25_excerpt_fixed_count` and `merged_bm25_excerpt_selected_count` put BM25 search into the Grep Token Reduction
agent, return an excerpt around the matching words of each passage with the line that excerpt starts on, and drop the turn
limit. Display names: BM25 Token Reduction, BM25 Token Reduction with model-selected count. Tuned on the validation questions.
`merged_bm25_token_reduction` and `merged_bm25_selected_count` are the first versions of these two, before that tuning.

The agent loop, the read and list_files tools and the call budget stay those of `merged_grep`. Search takes a query and
returns the five highest-ranked passages from the chunks plain RAG uses: BM25 alone, or BM25 and embedding merged by
reciprocal rank fusion. A passage longer than PASSAGE_LIMIT says so and names the line to read from.
"""

import re
from fnmatch import fnmatch
from pathlib import Path

import bm25s

from talk2.answer_format import ANSWER_INSTRUCTIONS
from talk2.harness_tools import CorpusTools
from talk2.retrieval_index import RECIPROCAL_RANK_CONSTANT, TOP_K, load_hybrid_retriever
from talk2.structured_tools import StructuredCorpusTools

PASSAGE_LIMIT = 1000
MAXIMUM_PASSAGE_COUNT = 10
EXCERPT_LIMIT = 600
EXCERPT_LEAD = 120

RANKED_SEARCH = {"type": "function", "function": {
    "name": "search",
    "description": ("Search every file in the corpus. Returns the five passages that best match the query, each with "
                    "its file and starting line. Results are ranked, so a term missing from them may still occur in "
                    "the corpus."),
    "parameters": {"type": "object", "properties": {
        "query": {"type": "string", "description": "Words to look for. Names, numbers and terms from the question work well."},
        "path_glob": {"type": "string", "description": "Restrict to files matching this glob, e.g. 'cuad/**' or "
                      "'**/christian_dior_2023.md'. Default: all files."},
    }, "required": ["query"]}}}

RANKED_SYSTEM_PROMPT = """You answer questions about a document collection stored as text files. You cannot see the files
directly; use the tools. search returns the passages that best match a query, ranked, so it cannot show that something is
absent. Read around the passages before answering. When you have the answer, stop calling tools and reply with the final
JSON.
""" + ANSWER_INSTRUCTIONS


class RankedCorpusTools(CorpusTools):
    search_definition = RANKED_SEARCH
    search_parameters = ("query", "path_glob")
    system_prompt = RANKED_SYSTEM_PROMPT

    def __init__(self, root: Path, suffix: str, corpus: str, use_embedding: bool, think: bool = False):
        super().__init__(root, suffix)
        self.think = think
        self.suffix_text = suffix
        self.use_embedding = use_embedding
        self.retriever = load_hybrid_retriever(corpus)
        self.nodes = self.retriever.nodes
        self.vector_retriever = self.retriever.index_for_all_nodes(len(self.nodes)) if use_embedding else None
        self.texts = {}

    def path_of(self, node) -> str:
        return f"{node.metadata['file_path']}{self.suffix_text}"

    def start_line(self, path: str, chunk: str) -> int:
        text = self.texts.setdefault(path, (self.root / path).read_text(errors="replace").replace("\f", "\n\n"))
        position = text.find(chunk[:120])
        return text.count("\n", 0, position) + 1 if position >= 0 else 1

    def ranked_nodes(self, query: str, path_glob: str, count: int = TOP_K) -> list[tuple[str, float]]:
        allowed = {n.node_id for n in self.nodes if not path_glob or fnmatch(self.path_of(n), path_glob) or fnmatch(self.path_of(n), f"**/{path_glob}")}
        positions, scores = self.retriever.keyword_index.retrieve(bm25s.tokenize([query], stopwords="en", show_progress=False),
                                                                  k=len(self.nodes), show_progress=False)
        keyword = [self.nodes[int(p)].node_id for p, s in zip(positions[0], scores[0]) if s > 0]
        if not self.use_embedding:
            return [(node_id, 1.0) for node_id in keyword if node_id in allowed][:count]
        rankings = [[node_id for node_id in keyword if node_id in allowed][:20],
                    [n.node.node_id for n in self.vector_retriever.retrieve(query) if n.node.node_id in allowed][:20]]
        fused: dict[str, float] = {}
        for ranking in rankings:
            for rank, node_id in enumerate(ranking, 1):
                fused[node_id] = fused.get(node_id, 0.0) + 1 / (RECIPROCAL_RANK_CONSTANT + rank)
        return sorted(fused.items(), key=lambda item: item[1], reverse=True)[:count]

    def search(self, query: str, path_glob: str = "") -> str:
        best = self.ranked_nodes(query, path_glob)
        if not best:
            return f"No passages match {query!r}."
        by_id = {n.node_id: n for n in self.nodes}
        passages = []
        for node_id, _ in best:
            node = by_id[node_id]
            path, chunk = self.path_of(node), node.get_content()
            line = self.start_line(path, chunk)
            note = f"\n[passage cut at {PASSAGE_LIMIT} of {len(chunk)} characters; read from line {line} for the rest]" if len(chunk) > PASSAGE_LIMIT else ""
            passages.append(f"{path}:{line}:\n{chunk[:PASSAGE_LIMIT]}{note}")
        return "\n\n".join(passages)


FIXED_COUNT_SEARCH = {"type": "function", "function": {
    "name": "search",
    "description": ("Search the corpus with BM25. Returns the five passages that best match the words of the query, each "
                    "with its file, line and page. Results are ranked, so a term missing from them may still occur in "
                    "the corpus."),
    "parameters": {"type": "object", "properties": {
        "query": {"type": "string", "description": "Words to look for. Names, numbers and terms from the question work well."},
        "file": {"type": "string", "description": "One path from the catalogue, or a glob such as 'cuad/**'. Default: all files."},
    }, "required": ["query"]}}}

SELECTED_COUNT_SEARCH = {"type": "function", "function": {
    "name": "search",
    "description": ("Search the corpus with BM25. Returns the passages that best match the words of the query, each with "
                    "its file, line and page. Results are ranked, so a term missing from them may still occur in the "
                    "corpus."),
    "parameters": {"type": "object", "properties": {
        "query": FIXED_COUNT_SEARCH["function"]["parameters"]["properties"]["query"],
        "file": FIXED_COUNT_SEARCH["function"]["parameters"]["properties"]["file"],
        "passage_count": {"type": "integer", "minimum": 1, "maximum": MAXIMUM_PASSAGE_COUNT,
                          "description": ("How many passages to return. Use 2 for one fact in a known file, 5 when unsure, "
                                          "and 10 when several documents may match or an earlier search missed.")},
    }, "required": ["query"]}}}


EXCERPT_DESCRIPTION = ("Search the corpus with BM25. Returns an excerpt around the matching words of each best passage, with "
                       "its file, the line the excerpt starts on, and its page. Results are ranked, so a term missing from "
                       "them may still occur in the corpus.")
EXCERPT_FIXED_COUNT_SEARCH = {"type": "function", "function": {**FIXED_COUNT_SEARCH["function"], "description": EXCERPT_DESCRIPTION}}
EXCERPT_SELECTED_COUNT_SEARCH = {"type": "function", "function": {
    "name": "search", "description": EXCERPT_DESCRIPTION,
    "parameters": {"type": "object", "properties": {
        "query": FIXED_COUNT_SEARCH["function"]["parameters"]["properties"]["query"],
        "file": FIXED_COUNT_SEARCH["function"]["parameters"]["properties"]["file"],
        "passage_count": {"type": "integer", "minimum": 1, "maximum": MAXIMUM_PASSAGE_COUNT,
                          "description": ("How many excerpts to return. Ask for few when the query holds an exact string or "
                                          "names one known file, and for more when several documents may match.")},
    }, "required": ["query"]}}}


def excerpt_bounds(chunk: str, query: str, limit: int = EXCERPT_LIMIT) -> tuple[int, int]:
    """Choose the window of the passage holding the most distinct query words. Longer words and numbers weigh more."""
    if len(chunk) <= limit:
        return 0, len(chunk)
    lowered = chunk.lower()
    terms = {term for term in re.findall(r"[^\W_]+(?:[.,][0-9]+)*", query.lower()) if len(term) > 1}
    weights = {term: len(term) + (10 if any(character.isdigit() for character in term) else 0) for term in terms}
    hits = sorted((match.start(), term) for term in terms for match in re.finditer(re.escape(term), lowered))
    best_start, best_weight = 0, 0
    for position, _ in hits:
        start = max(0, position - EXCERPT_LEAD)
        weight = sum(weights[term] for term in {term for at, term in hits if start <= at < start + limit})
        if weight > best_weight:
            best_start, best_weight = start, weight
    line_start = chunk.rfind("\n", 0, best_start) + 1
    if best_start - line_start <= EXCERPT_LEAD:
        best_start = line_start
    return best_start, min(len(chunk), best_start + limit)


class RankedStructuredTools(StructuredCorpusTools):
    """The tools of Grep Token Reduction with BM25 search in place of grep."""

    def __init__(self, root: Path, suffix: str, corpus: str, selected_count: bool, excerpts: bool = False):
        super().__init__(root, suffix, show_long_lines=True)
        self.excerpts = excerpts
        self.ranking = RankedCorpusTools(root, suffix, corpus, use_embedding=False)
        self.selected_count = selected_count
        self.search_definition = {(False, False): FIXED_COUNT_SEARCH, (True, False): SELECTED_COUNT_SEARCH,
                                  (False, True): EXCERPT_FIXED_COUNT_SEARCH, (True, True): EXCERPT_SELECTED_COUNT_SEARCH}[selected_count, excerpts]

    def passage_count(self, arguments: dict) -> int:
        if not self.selected_count:
            return TOP_K
        try:
            return max(1, min(MAXIMUM_PASSAGE_COUNT, int(arguments.get("passage_count", TOP_K))))
        except (TypeError, ValueError):
            return TOP_K

    def search_from_arguments(self, arguments: dict) -> str:
        query = str(arguments["query"])
        path = str(arguments.get("file") or arguments.get("path") or "")
        best = self.ranking.ranked_nodes(query, path, self.passage_count(arguments))
        if not best:
            return (f"0 matches for {query!r}. This does not yet show absence. Try one distinctive word, a synonym, or the "
                    f"wording of the document's language (corpus languages: {', '.join(self.languages)}).")
        by_id = {node.node_id: node for node in self.ranking.nodes}
        passages = []
        for node_id, _ in best:
            node = by_id[node_id]
            location, chunk = self.ranking.path_of(node), node.get_content()
            line = self.ranking.start_line(location, chunk)
            if self.excerpts:
                start, end = excerpt_bounds(chunk, query)
                line += chunk.count("\n", 0, start)
                pages = self.page_of_each_line(self.file_lines(location) or [])
                page = pages[line - 1] if line <= len(pages) else "?"
                note = (f"\n[excerpt of {end - start} from {len(chunk)} characters; read from line {line} for the rest]"
                        if end - start < len(chunk) else "")
                passages.append(f"{location}:{line} [page {page}]:\n{chunk[start:end]}{note}")
                continue
            pages = self.page_of_each_line(self.file_lines(location) or [])
            page = pages[line - 1] if line <= len(pages) else "?"
            note = (f"\n[passage cut at {PASSAGE_LIMIT} of {len(chunk)} characters; read from line {line} for the rest]"
                    if len(chunk) > PASSAGE_LIMIT else "")
            passages.append(f"{location}:{line} [page {page}]:\n{chunk[:PASSAGE_LIMIT]}{note}")
        return "\n\n".join(passages)
