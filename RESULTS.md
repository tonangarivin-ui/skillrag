# Retrieval results

Corpus: 635 markdown files (194 `SKILL.md` plus references), 5,106 chunks after heading-based chunking with a 10-chunk-per-document cap. Index builds in ~1.1 s in-process; no service, no vector store.

Command: `python3 rag.py eval`

## Run 1 — BM25 over body text only

```
top-1 8/12 = 67% | top-3 11/12 = 92%
```

Misses:

- `linux-server-admin` — the question mentions "systemd" and "tsx", and the chunk that actually explains the restart loop did not make the top 3. Neighbouring words pulled other skills up: the corpus has dozens of documents that mention systemd.
- `buffer-api`, `remotion-video`, `songwriting` — right skill family, wrong document inside it (top-3 landed on a sibling skill in the same category).

## Run 2 — heading and path tokens weighted ×2

Same 12 questions, same chunking, one change: the tokens of the document's heading and its file path are counted twice, because they state what the document is for, not just what it happens to contain.

```
top-1 8/12 = 67% | top-3 12/12 = 100%
```

Top-1 is unchanged: on four questions a sibling document inside the correct skill still wins. Top-3 going to 100% is the number that matters for a RAG system, because the answering model receives all three excerpts.

## What this says, honestly

- Weighting metadata beat any tuning of k1/b that I tried, and it cost one line.
- Top-1 plateauing at 67% is the signal for the next real improvement: reranking the top-5 with a cross-encoder or an LLM judge. Not done here — this repo stops at measuring the cheap baseline.
- Twelve questions cannot prove generalisation. They guard against regression, and every one of them is a question I have actually asked my own agent.

## Blocked: vector retrieval

`gemini-embedding-001` via the local gateway returns HTTP 429 (Google quota) after a few hundred chunks; the full index needs ~640 batched calls. The vector path is implemented (`index --vectors`), retries with backoff, and resumes from a partial file, but no full run exists yet — so there is no vector score in this document. An unmeasured retriever does not get a row in the table.
