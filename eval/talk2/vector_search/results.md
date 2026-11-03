# Local vector search on the talk's own vectors

`scripts/talk2/talk2/vector_store_comparison.py`, 3 Nov 2026, 32 GB Mac. 200 queries: stored vectors with small noise added,
because this measures the index and not the embedding. Recall is against the exact top 5.

| Vectors | Option | Build | Median query | p99 query | Recall@5 |
|---|---|---|---|---|---|
| 1,219 (unseen corpus) | FAISS IndexFlatIP | 0 s | 0.13 ms | 0.22 ms | 1.000 |
| 1,219 | Voyager HNSW (M 16, ef 64) | 0.05 s | 0.10 ms | 0.15 ms | 0.968 |
| 1,219 | LanceDB, exact | 0.08 s | 3.5 ms | 6.0 ms | 1.000 |
| 1,219 | Chroma HNSW | 2.2 s | 2.4 ms | 11.4 ms | 0.980 |
| 43,454 (605 PDFs) | FAISS IndexFlatIP | 0 s | 2.2 ms | 3.6 ms | 1.000 |
| 43,454 | Voyager HNSW (M 16, ef 64) | 9.7 s | 0.80 ms | 1.6 ms | 0.978 |
| 43,454 | LanceDB, exact | 1.0 s | 25 ms | 155 ms | 1.000 |

A question takes about 15 seconds through the linear pipe, so the vector search is under 0.01% of the wall time at this scale.
FAISS stays: exact results at this size mean a retrieval miss is the embedding's miss, not the index's. Voyager becomes the
choice when exact search starts to cost more than the lost recall, which these two sizes do not reach.
