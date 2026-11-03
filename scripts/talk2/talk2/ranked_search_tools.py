"""Configurations `merged_bm25_loop` and `merged_fusion_loop`: the grep agent with its search tool replaced.

The agent loop, the read and list_files tools and the call budget stay those of `merged_grep`. Search takes a query and
returns the five highest-ranked passages from the chunks plain RAG uses: BM25 alone, or BM25 and embedding merged by
reciprocal rank fusion. A passage longer than PASSAGE_LIMIT says so and names the line to read from.
"""

from fnmatch import fnmatch
from pathlib import Path

import bm25s

from talk2.answer_format import ANSWER_INSTRUCTIONS
from talk2.harness_tools import CorpusTools
from talk2.retrieval_index import RECIPROCAL_RANK_CONSTANT, TOP_K, load_hybrid_retriever

PASSAGE_LIMIT = 1000

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

    def __init__(self, root: Path, suffix: str, corpus: str, use_embedding: bool):
        super().__init__(root, suffix)
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

    def ranked_nodes(self, query: str, path_glob: str) -> list[tuple[str, float]]:
        allowed = {n.node_id for n in self.nodes if not path_glob or fnmatch(self.path_of(n), path_glob) or fnmatch(self.path_of(n), f"**/{path_glob}")}
        positions, scores = self.retriever.keyword_index.retrieve(bm25s.tokenize([query], stopwords="en", show_progress=False),
                                                                  k=len(self.nodes), show_progress=False)
        keyword = [self.nodes[int(p)].node_id for p, s in zip(positions[0], scores[0]) if s > 0]
        if not self.use_embedding:
            return [(node_id, 1.0) for node_id in keyword if node_id in allowed][:TOP_K]
        rankings = [[node_id for node_id in keyword if node_id in allowed][:20],
                    [n.node.node_id for n in self.vector_retriever.retrieve(query) if n.node.node_id in allowed][:20]]
        fused: dict[str, float] = {}
        for ranking in rankings:
            for rank, node_id in enumerate(ranking, 1):
                fused[node_id] = fused.get(node_id, 0.0) + 1 / (RECIPROCAL_RANK_CONSTANT + rank)
        return sorted(fused.items(), key=lambda item: item[1], reverse=True)[:TOP_K]

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
