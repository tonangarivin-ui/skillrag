#!/usr/bin/env python3
"""skillrag — retrieval over the agent's own skill documents.

    python3 rag.py index              # chunk + build the BM25 index
    python3 rag.py index --vectors    # also embed (needs gateway embedding quota)
    python3 rag.py ask "question"     # retrieve, then answer with citations
    python3 rag.py eval               # retrieval hit-rate on eval.json

Default retrieval is BM25: no quota, no network, deterministic. Vector mode is
opt-in because the gateway's embedding provider enforces a Google quota that a
5,000-chunk index blows straight through.
"""
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

SKILLS = Path(os.environ.get("SKILLRAG_CORPUS", Path.home() / ".hermes/skills"))
HERE = Path(__file__).resolve().parent
CHUNKS = HERE / "chunks.json"
BM25 = HERE / "bm25.json"
VECTORS = HERE / "vectors.json"
PARTIAL = HERE / "vectors.partial.jsonl"
BASE = os.environ.get("SKILLRAG_BASE", "http://127.0.0.1:20128/v1")
EMBED_MODEL = os.environ.get("SKILLRAG_EMBED_MODEL", "gemini-embedding-001")
CHAT_MODEL = os.environ.get("SKILLRAG_CHAT_MODEL", "river/deepseek-ai/DeepSeek-V4.1-Flash-1017")
MAX_CHARS = 1100
MIN_CHARS = 120
CHUNKS_PER_DOC = 10
DIMS = 768

STOP = set(
    """a an and are as at be but by for from has have how i if in into is it its of on or that the their
    then there these they this to was were what when where which who why will with you your do does""".split()
)


# ---------- corpus ----------

def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9_]+", text.lower()) if len(t) > 1 and t not in STOP]


def chunks_of(path: Path) -> list[dict]:
    """Split a markdown document on headings, then on paragraph boundaries."""
    text = path.read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.S)  # frontmatter
    if not text.strip():
        return []
    out: list[dict] = []
    for part in re.split(r"\n(?=#{1,3} )", text):
        heading = part.lstrip("#").split("\n", 1)[0].strip()[:90]
        buf = ""
        for para in part.split("\n\n"):
            if len(buf) + len(para) > MAX_CHARS and len(buf) >= MIN_CHARS:
                out.append({"heading": heading, "text": buf.strip()})
                buf = para
            else:
                buf = f"{buf}\n\n{para}" if buf else para
        if len(buf.strip()) >= MIN_CHARS:
            out.append({"heading": heading, "text": buf.strip()})
    keep = out[:CHUNKS_PER_DOC]  # ponytail: cap per doc; raise if long skills lose their tail answers
    return [{"path": str(path.relative_to(SKILLS)), "heading": c["heading"], "text": c["text"]} for c in keep]


# ---------- BM25 ----------

def build_bm25(chunks: list[dict]) -> dict:
    postings: dict[str, list[list[int]]] = {}
    lengths: list[int] = []
    for i, c in enumerate(chunks):
        # heading and path terms describe the document; weight them like they matter
        counts = Counter(tokens(c["text"]) + tokens(c["heading"]) * 2 + tokens(c["path"]) * 2)
        lengths.append(sum(counts.values()))
        for term, tf in counts.items():
            postings.setdefault(term, []).append([i, tf])
    return {"postings": postings, "lengths": lengths, "avgdl": sum(lengths) / max(len(lengths), 1)}


def bm25_search(question: str, k: int = 5, index: dict | None = None, chunks: list[dict] | None = None) -> list[tuple[float, dict]]:
    index = index or json.loads(BM25.read_text())
    chunks = chunks or json.loads(CHUNKS.read_text())
    postings, lengths, avgdl = index["postings"], index["lengths"], index["avgdl"]
    n = len(lengths)
    k1, b = 1.5, 0.75
    scores: dict[int, float] = {}
    for term in set(tokens(question)):
        posting = postings.get(term)
        if not posting:
            continue
        idf = math.log(1 + (n - len(posting) + 0.5) / (len(posting) + 0.5))
        for doc, tf in posting:
            denom = tf + k1 * (1 - b + b * lengths[doc] / avgdl)
            scores[doc] = scores.get(doc, 0.0) + idf * tf * (k1 + 1) / denom
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
    return [(round(score, 3), chunks[doc]) for doc, score in ranked]


# ---------- vectors (optional) ----------

def api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    env = Path.home() / ".hermes/.env"
    m = re.search(r"OPENAI_API_KEY\s*=\s*\"?([^\s\"\n#]+)", env.read_text()) if env.exists() else None
    if not m:
        sys.exit("no OPENAI_API_KEY in env or ~/.hermes/.env")
    return m.group(1)


def post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {api_key()}", "content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())


def shrink(vec: list[float]) -> list[float]:
    head = vec[:DIMS]
    n = math.sqrt(sum(x * x for x in head)) or 1.0
    return [x / n for x in head]


def embed_batch(batch: list[str], attempts: int = 5) -> list[dict]:
    """The gateway rate-limits bursts; back off instead of dropping work."""
    delay = 3.0
    for n in range(attempts):
        try:
            return post("/embeddings", {"model": EMBED_MODEL, "input": batch})["data"]
        except urllib.error.HTTPError as e:
            if e.code != 429 or n == attempts - 1:
                raise
            print(f"  429, sleeping {delay:.0f}s", file=sys.stderr)
            time.sleep(delay)
            delay *= 2
    raise SystemExit("unreachable")


def build_vectors(chunks: list[dict]) -> None:
    done = sum(1 for _ in PARTIAL.open()) if PARTIAL.exists() else 0
    if done:
        print(f"  resuming after {done} chunks", file=sys.stderr)
    with PARTIAL.open("a") as out:
        for i in range(done, len(chunks), 8):
            batch = chunks[i : i + 8]
            for d in embed_batch([f"{c['heading']}\n{c['text']}" for c in batch]):
                out.write(json.dumps([round(x, 5) for x in shrink(d["embedding"])]) + "\n")
            out.flush()
            print(f"  embedded {min(i + 8, len(chunks))}/{len(chunks)}", file=sys.stderr)
            time.sleep(0.35)
    VECTORS.write_text(json.dumps([json.loads(line) for line in PARTIAL.open()]))
    PARTIAL.unlink()


def vector_search(question: str, k: int = 5) -> list[tuple[float, dict]]:
    chunks = json.loads(CHUNKS.read_text())
    vectors = json.loads(VECTORS.read_text())
    q = shrink(embed_batch([question])[0]["embedding"])
    scored = []
    for v, c in zip(vectors, chunks):
        dot = sum(a * b for a, b in zip(q, v))
        scored.append((round(dot, 3), c))
    return sorted(scored, key=lambda t: -t[0])[:k]


# ---------- commands ----------

def cmd_index(with_vectors: bool) -> None:
    files = sorted(SKILLS.rglob("*.md"))
    chunks: list[dict] = []
    for f in files:
        chunks.extend(chunks_of(f))
    CHUNKS.write_text(json.dumps(chunks, ensure_ascii=False))
    index = build_bm25(chunks)
    BM25.write_text(json.dumps(index))
    print(f"{len(files)} documents -> {len(chunks)} chunks, {len(index['postings'])} terms")
    if with_vectors:
        build_vectors(chunks)


def search(question: str, k: int = 5) -> list[tuple[float, dict]]:
    if not CHUNKS.exists():
        sys.exit("run `python3 rag.py index` first")
    if VECTORS.exists():
        return vector_search(question, k)
    return bm25_search(question, k)


def cmd_ask(question: str) -> None:
    hits = search(question, k=6)
    if not hits:
        sys.exit("no chunk matched — try different wording")
    context = "\n\n".join(f"[{i + 1}] {c['path']} — {c['heading']}\n{c['text']}" for i, (_, c) in enumerate(hits))
    res = post(
        "/chat/completions",
        {
            "model": CHAT_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "Answer only from the numbered excerpts. Cite as [n]. If the excerpts do not answer it, say so plainly.",
                },
                {"role": "user", "content": f"Question: {question}\n\n{context}"},
            ],
            "temperature": 0.2,
            "stream": False,
        },
    )
    print(res["choices"][0]["message"]["content"].strip())
    print("\nsources:")
    for score, c in hits:
        print(f"  {score}  {c['path']} — {c['heading']}")


def cmd_eval() -> None:
    cases = json.loads((HERE / "eval.json").read_text())
    index = json.loads(BM25.read_text())
    chunks = json.loads(CHUNKS.read_text())
    top1 = top3 = 0
    for case in cases:
        hits = bm25_search(case["q"], k=3, index=index, chunks=chunks)
        paths = [c["path"] for _, c in hits]
        want = case["expect"]
        hit1 = bool(paths) and want in paths[0]
        hit3 = any(want in p for p in paths)
        top1 += hit1
        top3 += hit3
        print(f"{'ok  ' if hit3 else 'MISS'} 1:{'y' if hit1 else 'n'} want={want:26} got={[p.split('/')[0] for p in paths]}")
    n = len(cases)
    print(f"\ntop-1 {top1}/{n} = {top1 / n:.0%} | top-3 {top3}/{n} = {top3 / n:.0%}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "index":
        cmd_index("--vectors" in sys.argv)
    elif cmd == "ask" and len(sys.argv) > 2:
        cmd_ask(" ".join(sys.argv[2:]))
    elif cmd == "eval":
        cmd_eval()
    else:
        print(__doc__)
