"""Compare local vector search options on real talk vectors: build time, query latency, recall of the exact top 5."""
import gc, json, os, shutil, sys, tempfile, time, statistics
from pathlib import Path
import numpy as np, faiss

def run(path, name):
    index = faiss.read_index(str(path))
    n, d = index.ntotal, index.d
    vectors = index.reconstruct_n(0, n).astype("float32")
    rng = np.random.default_rng(42)
    picks = rng.choice(n, size=min(200, n), replace=False)
    queries = vectors[picks] + rng.normal(0, 0.02, (len(picks), d)).astype("float32")
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    exact = index.search(queries, 5)[1]
    results = []
    def measure(label, build, search):
        started = time.perf_counter(); store = build(); built = time.perf_counter() - started
        times, recall = [], []
        for query, truth in zip(queries, exact):
            t = time.perf_counter(); ids = search(store, query); times.append((time.perf_counter() - t) * 1000)
            recall.append(len(set(map(int, ids)) & set(map(int, truth))) / 5)
        results.append((label, built, statistics.median(times), sorted(times)[int(.99 * len(times))], statistics.mean(recall)))
        print(f"{name} n={n} d={d} {label:24s} build {built:8.3f} s  median {results[-1][2]:8.3f} ms  p99 {results[-1][3]:8.3f} ms  recall@5 {results[-1][4]:.3f}", flush=True)
    measure("faiss IndexFlatIP", lambda: index, lambda s, q: s.search(q[None], 5)[1][0])
    try:
        import voyager
        def build():
            v = voyager.Index(voyager.Space.InnerProduct, num_dimensions=d, M=16, ef_construction=200)
            v.add_items(vectors); return v
        measure("voyager HNSW", build, lambda s, q: s.query(q, k=5, query_ef=64)[0])
    except Exception as e: print("voyager failed", type(e).__name__, str(e)[:200])
    try:
        import lancedb, pyarrow as pa
        folder = tempfile.mkdtemp()
        def build():
            db = lancedb.connect(folder)
            table = pa.table({"id": pa.array(range(n)), "vector": pa.FixedSizeListArray.from_arrays(pa.array(vectors.flatten()), d)})
            return db.create_table("chunks", table)
        measure("lancedb exact", build, lambda s, q: s.search(q).metric("dot").limit(5).select(["id"]).to_arrow()["id"].to_pylist())
    except Exception as e: print("lancedb failed", type(e).__name__, str(e)[:200])
    try:
        import chromadb
        folder2 = tempfile.mkdtemp()
        def build():
            c = chromadb.PersistentClient(path=folder2).create_collection("chunks", metadata={"hnsw:space": "ip"})
            for s in range(0, n, 2000):
                e = min(n, s + 2000); c.add(ids=[str(i) for i in range(s, e)], embeddings=vectors[s:e].tolist())
            return c
        measure("chroma HNSW", build, lambda s, q: s.query(query_embeddings=[q.tolist()], n_results=5, include=[])["ids"][0])
    except Exception as e: print("chroma failed", type(e).__name__, str(e)[:200])

for item in sys.argv[1:]:
    label, path = item.split("=", 1)
    run(Path(path), label)
