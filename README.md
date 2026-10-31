# skillrag

Retrieval-augmented answers over my own agent skill library: **635 markdown documents → 5,106 chunks**, every one of them written by me.

The interesting question is not "can an LLM answer from context". It is **can plain-language questions find the one document that answers them?** Everything else follows from retrieval quality, so that is what this repo measures.

## Results (reproduce with `python3 rag.py eval`)

| retriever | top-1 | top-3 | index build |
|---|---|---|---|
| BM25, body only | 67% (8/12) | 92% (11/12) | 1.1 s |
| **BM25 + heading/path weighting** | 67% (8/12) | **100% (12/12)** | 1.1 s |

The 12 evaluation questions are deliberately cross-domain — stock-submission limits, a systemd restart loop, Buffer's GraphQL API, Suno prompts, Polymarket order books — so a lucky keyword cannot carry the score. Details and the failing case: `RESULTS.md`.

## How it works

```
skills/**/*.md ─► chunk on headings (~1,100 chars, 10 chunks/doc max) ─► BM25 index
                                                                          │
question ─────────────────────────► same tokenizer ─► top-k chunks ───────┴─► LLM answer with [n] citations
```

- **Chunking**: split on `#`/`##`/`###`, then paragraph boundaries. Frontmatter dropped, heading kept as context.
- **Retrieval**: BM25 (k1=1.5, b=0.75) over an inverted index built in-process in about a second. Heading and path tokens are weighted double, because they describe what the document is *for* — that single change took top-3 from 92% to 100%.
- **Answering**: the retrieved chunks go to a chat model with a strict instruction: answer only from the excerpts, cite as `[n]`, and say so when the excerpts do not answer the question.
- **Citations**: the CLI prints the source path and heading for every chunk used, so a wrong answer is traceable to the retrieval step, not hidden in the prose.

## Vector mode (optional, currently blocked)

`python3 rag.py index --vectors` embeds chunks with `gemini-embedding-001` through a local OpenAI-compatible gateway, truncated to 768 dimensions and renormalised. It is opt-in and **not** the default, for a boring reason: the gateway's Google quota returns HTTP 429 after a few hundred chunks, and a 5,106-chunk index needs ~640 batched calls. The code retries with exponential backoff and resumes from `vectors.partial.jsonl`, so it will finish when quota allows.

Shipping the lexical retriever that works beats shipping a vector index that half-exists. When the quota allows a full run, the two retrievers will be compared on the same 12 questions.

## HTTP endpoint + n8n workflow

The retriever is also a service, so automation can call it (`skillrag.service`, `127.0.0.1:3310`):

```bash
curl -s -X POST http://127.0.0.1:3310/ask \
  -H 'content-type: application/json' \
  -d '{"question":"What are the Adobe Stock video duration limits?"}'
# {"answer":"5 to 60 seconds [2]...","sources":[{"path":"creative/microstock-submission/references/adobe-stock.md",...}]}
```

`n8n-workflow.json` is the same thing as a workflow: **Webhook POST /webhook/skillrag → HTTP Request to skillrag → Respond**. Import on an existing n8n instance with:

```bash
docker cp n8n-workflow.json n8n:/tmp/wf.json
docker exec n8n n8n import:workflow --input=/tmp/wf.json
docker exec n8n n8n update:workflow --id=<id> --active=true
docker restart n8n
```

Verified end to end: `POST http://127.0.0.1:5678/webhook/skillrag {"question": "..."}` returns the answer and its sources, HTTP 200.

Live instances:

| service | URL | access |
|---|---|---|
| skillrag API | `https://rag.darklabs.tech` | `POST /ask` needs `x-api-key` (the proxy sets `X-Forwarded-For`, direct loopback calls do not, so n8n needs no token) |
| n8n editor | `https://flow.darklabs.tech` | the existing n8n account |
| n8n webhook | `https://flow.darklabs.tech/webhook/...` | **blocked at nginx (403)** — a public webhook would proxy straight to the LLM |

The API token lives in `.token` (gitignored, mode 600) or `SKILLRAG_TOKEN`; requests through the proxy without it get `401`.

## Commands

```bash
python3 rag.py index                                   # build chunks.json + bm25.json (~1 s)
python3 rag.py ask "how do I expose a local service without opening ports?"
python3 rag.py eval                                    # retrieval hit-rate vs eval.json
python3 rag.py index --vectors                         # also embed (needs embedding quota)
```

`chunks.json`, `bm25.json` and `vectors*.json*` are build artifacts and are not committed.

Retrieval is wording-sensitive: the question *"why does a systemd unit using tsx keep restarting"* retrieves three plausible systemd documents but not the one that explains it, while *"why does a systemd unit using ExecStart with tsx restart loop"* does. That is a vocabulary-gap miss, and it is the case for a reranker or query expansion — not something the current scorer can fix.

## What this does not do

- No reranker, no query rewriting, no hybrid fusion. Each is a real improvement; the cheap baseline had to be measured first, and the numbers above are what "before" looks like.
- 12 questions is a smoke test, not a benchmark. It catches regressions, it does not prove generalisation.
- The corpus is one person's technical writing, so retrieval works over a domain with consistent vocabulary. A PDF dump of arbitrary documents would score worse.
