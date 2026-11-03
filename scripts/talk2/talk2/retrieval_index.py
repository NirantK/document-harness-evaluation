"""Plain-RAG index: LlamaIndex SentenceSplitter(512, 50) + Qwen3-Embedding-0.6B (served by Ollama) + FAISS IndexFlatIP.

The same builder indexes the PyMuPDF text corpus (configuration `rag`) and the Markdown corpus (`markdown_rag`).
Hybrid retrieval (`text_hybrid`, `markdown_hybrid`) adds BM25 (bm25s) over the same chunks and fuses the two
rankings by reciprocal rank fusion.
"""

from pathlib import Path

import bm25s
import faiss
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from tokenizers import Tokenizer

from talk2.corpus_assembly import CACHE_ROOT, CORPUS_NAME, CORPUS_PDF

EMBEDDING_MODEL = "qwen3-embedding:0.6b"
TOKENIZER_REPOSITORY = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_DIMENSION = 1024
QUERY_INSTRUCTION = "Instruct: Given a question, retrieve passages that answer the question\nQuery: "
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K = 5
FUSION_CANDIDATES = 20
RECIPROCAL_RANK_CONSTANT = 60
CORPUS_ROOTS = {"text": CORPUS_PDF.parent / "text", "markdown": CORPUS_PDF.parent / "markdown",
                "markdown_merged": CORPUS_PDF.parent / "markdown_merged"}
INDEX_ROOT = CACHE_ROOT if CORPUS_NAME == "full" else CACHE_ROOT / CORPUS_NAME


def configure_embedding() -> None:
    Settings.embed_model = OllamaEmbedding(
        model_name=EMBEDDING_MODEL, embed_batch_size=32, query_instruction=QUERY_INSTRUCTION, text_instruction="",
        ollama_additional_kwargs={"num_ctx": 2048},
    )
    tokenizer = Tokenizer.from_pretrained(TOKENIZER_REPOSITORY)
    Settings.tokenizer = lambda text: tokenizer.encode(text, add_special_tokens=False).ids


def load_documents(corpus: str) -> list[Document]:
    root = CORPUS_ROOTS[corpus]
    suffix = ".txt" if corpus == "text" else ".md"
    documents = []
    for path in sorted(root.rglob(f"*{suffix}")):
        relative = str(path.relative_to(root).with_suffix(""))
        documents.append(Document(text=path.read_text().replace("\f", "\n\n"), metadata={"file_path": relative}))
    return documents


def build_index(corpus: str) -> int:
    configure_embedding()
    documents = load_documents(corpus)
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, tokenizer=Settings.tokenizer)
    nodes = splitter.get_nodes_from_documents(documents, show_progress=True)
    vector_store = FaissVectorStore(faiss_index=faiss.IndexFlatIP(EMBEDDING_DIMENSION))
    storage = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes, storage_context=storage, show_progress=True)
    index.storage_context.persist(persist_dir=str(INDEX_ROOT / f"index_{corpus}"))
    return len(nodes)


def load_retriever(corpus: str):
    configure_embedding()
    directory = str(INDEX_ROOT / f"index_{corpus}")
    vector_store = FaissVectorStore.from_persist_dir(directory)
    storage = StorageContext.from_defaults(vector_store=vector_store, persist_dir=directory)
    return load_index_from_storage(storage).as_retriever(similarity_top_k=TOP_K)


class HybridRetriever:
    """BM25 and embedding rankings over the same chunks, fused by reciprocal rank fusion; returns the top TOP_K."""

    def __init__(self, index):
        self.index = index
        self.nodes = list(index.docstore.docs.values())
        self.vector_retriever = index.as_retriever(similarity_top_k=FUSION_CANDIDATES)
        self.keyword_index = bm25s.BM25()
        self.keyword_index.index(bm25s.tokenize([n.get_content() for n in self.nodes], stopwords="en",
                                                show_progress=False), show_progress=False)

    def index_for_all_nodes(self, count: int):
        """An embedding retriever that ranks every chunk, so a path filter can still return five passages."""
        return self.index.as_retriever(similarity_top_k=count)

    def retrieve(self, question: str) -> list[NodeWithScore]:
        positions, _ = self.keyword_index.retrieve(bm25s.tokenize([question], stopwords="en", show_progress=False),
                                                   k=min(FUSION_CANDIDATES, len(self.nodes)), show_progress=False)
        rankings = [[self.nodes[int(p)].node_id for p in positions[0]],
                    [n.node.node_id for n in self.vector_retriever.retrieve(question)]]
        fused: dict[str, float] = {}
        for ranking in rankings:
            for rank, node_id in enumerate(ranking, 1):
                fused[node_id] = fused.get(node_id, 0.0) + 1 / (RECIPROCAL_RANK_CONSTANT + rank)
        by_id = {n.node_id: n for n in self.nodes}
        best = sorted(fused, key=fused.get, reverse=True)[:TOP_K]
        return [NodeWithScore(node=by_id[node_id], score=fused[node_id]) for node_id in best]


def load_hybrid_retriever(corpus: str) -> HybridRetriever:
    configure_embedding()
    directory = str(INDEX_ROOT / f"index_{corpus}")
    vector_store = FaissVectorStore.from_persist_dir(directory)
    storage = StorageContext.from_defaults(vector_store=vector_store, persist_dir=directory)
    return HybridRetriever(load_index_from_storage(storage))
