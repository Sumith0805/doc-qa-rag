"""Repeatable evaluation for DocuMind.

    python -m eval.run_eval                 # retrieval + abstention only (no LLM key needed)
    python -m eval.run_eval --llm groq      # + answer correctness, citation correctness, latency
    python -m eval.run_eval --validate      # only check that dataset labels match the documents

Uses a throw-away temp database, so it never touches your real uploaded documents.
Every run is saved to eval/results/ and compared against the previous run.
"""
import argparse
import datetime
import json
import os
import re
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))


def norm(s):
    return re.sub(r"\s+", " ", s).lower()


def validate_dataset(dataset):
    """Catch wrong labels BEFORE blaming the model: every expected fact must really be on
    the expected page; every 'unsupported' question must really be absent from the docs."""
    from chunker import read_document

    texts = {}
    for rel in dataset["documents"]:
        p = HERE / rel
        texts[p.name] = {pg: norm(t) for pg, t in read_document(str(p), p.name)}
    errors = []
    for c in dataset["cases"]:
        for s in c.get("sources", []):
            pages = texts.get(s["filename"])
            if pages is None:
                errors.append(f"{c['id']}: unknown file {s['filename']}")
                continue
            if not any(norm(s["evidence"]) in pages.get(pg, "") for pg in s["pages"]):
                errors.append(f"{c['id']}: evidence '{s['evidence']}' not found on pages {s['pages']} of {s['filename']}")
        for bad in c.get("must_not_contain", []):
            for fn, pages in texts.items():
                if any(norm(bad) in t for t in pages.values()):
                    errors.append(f"{c['id']}: '{bad}' DOES appear in {fn}, so the question is answerable")
    return errors


def pct(x, n):
    return round(100.0 * x / n, 1) if n else None


def percentile(vals, p):
    if not vals:
        return None
    vals = sorted(vals)
    return vals[min(len(vals) - 1, int(round(p / 100 * (len(vals) - 1))))]


def source_match(hit_or_cite, source):
    """Lenient: right file and right page."""
    return hit_or_cite["filename"] == source["filename"] and (
        not source["pages"] or hit_or_cite["page"] in source["pages"]
    )


def evidence_match(hit_or_cite, source):
    """Strict: right file, right page AND the chunk text really contains the expected fact."""
    return source_match(hit_or_cite, source) and norm(source["evidence"]) in norm(hit_or_cite["text"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", choices=["none", "groq", "mock"], default=None)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--min-score", type=float, default=None)
    ap.add_argument("--label", default="run")
    ap.add_argument("--sleep", type=float, default=0.3, help="pause between LLM calls (rate limits)")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="documind_eval_")
    os.environ["DOCUMIND_DATA_DIR"] = tmp  # must be set BEFORE importing config
    import config  # noqa: E402

    dataset = json.loads((HERE / "dataset.json").read_text(encoding="utf-8"))
    errors = validate_dataset(dataset)
    if errors:
        print("DATASET LABEL ERRORS:")
        for e in errors:
            print("  -", e)
        sys.exit(2)
    print(f"Dataset OK: {len(dataset['cases'])} cases, labels verified against the documents.")
    if args.validate:
        shutil.rmtree(tmp, ignore_errors=True)
        return

    llm = args.llm or ("groq" if os.getenv("GROQ_API_KEY") else "none")
    config.LLM_MODE = "mock" if llm == "mock" else "groq"
    k = args.k or config.TOP_K
    min_score = config.MIN_SCORE if args.min_score is None else args.min_score

    import rag  # noqa: E402
    import store  # noqa: E402

    print(f"Indexing {len(dataset['documents'])} documents (first run downloads the embedding model)...")
    for rel in dataset["documents"]:
        p = HERE / rel
        rec = store.index_document(str(p), p.name)
        print(f"  {p.name}: {rec['pages']} pages, {rec['chunks']} chunks")

    results = []
    for n, c in enumerate(dataset["cases"], start=1):
        row = {"id": c["id"], "category": c["category"], "question": c["question"], "failures": []}
        try:
            t0 = time.perf_counter()
            hits = store.search(c["question"], k)
            row["retrieval_ms"] = int((time.perf_counter() - t0) * 1000)
            row["top_score"] = max((h["score"] for h in hits), default=0.0)

            if llm == "none":
                llm_fn = lambda prompt, h: "(retrieval-only run) [S1]"  # noqa: E731
            else:
                llm_fn = None
            t1 = time.perf_counter()
            res = rag.answer(c["question"], k=k, min_score=min_score,
                             search_fn=lambda q, kk: hits, llm_fn=llm_fn)
            row["total_ms"] = row["retrieval_ms"] + int((time.perf_counter() - t1) * 1000)
            row["abstained"] = res["abstained"]
            row["answer"] = res["answer"]
            row["retrieved"] = [{"filename": h["filename"], "page": h["page"], "score": h["score"]}
                                for h in res["retrieved"]]
            if llm != "none" and not res["abstained"]:
                time.sleep(args.sleep)

            if c["category"] == "unsupported":
                row["abstain_correct"] = res["abstained"]
                if not res["abstained"]:
                    row["failures"].append("hallucination: answered an unsupported question"
                                           if llm != "none" else "weak-evidence gate let it through")
            else:
                ranks = [i for i, h in enumerate(res["retrieved"], 1)
                         if any(evidence_match(h, s) for s in c["sources"])]
                row["page_hit"] = all(
                    any(source_match(h, s) for h in res["retrieved"]) for s in c["sources"])
                row["retrieval_hit"] = all(
                    any(evidence_match(h, s) for h in res["retrieved"]) for s in c["sources"])
                row["first_rank"] = ranks[0] if ranks else None
                row["wrongly_abstained"] = res["abstained"]
                if not row["retrieval_hit"]:
                    missing = [f"{s['filename']} (needs: {s['evidence']})" for s in c["sources"]
                               if not any(evidence_match(h, s) for h in res["retrieved"])]
                    row["failures"].append(f"retrieval miss: no retrieved chunk contains the fact -> {', '.join(missing)}")
                if res["abstained"]:
                    row["failures"].append(f"wrongly abstained (top score {row['top_score']})")
                if llm != "none" and not res["abstained"]:
                    low = norm(res["answer"])
                    ok = True
                    if c.get("answer_any"):
                        ok &= any(norm(a) in low for a in c["answer_any"])
                    if c.get("answer_all"):
                        ok &= all(norm(a) in low for a in c["answer_all"])
                    row["answer_correct"] = ok
                    row["citation_correct"] = bool(res["grounded"]) and all(
                        any(evidence_match(cit, s) for cit in res["citations"]) for s in c["sources"])
                    if not ok:
                        row["failures"].append("answer missing expected facts")
                    if not row["citation_correct"]:
                        row["failures"].append("citation missing/incorrect")
        except Exception as exc:  # keep going; record the error
            row["failures"].append(f"error: {exc}")
        results.append(row)
        print(f"[{n:02d}/{len(dataset['cases'])}] {c['id']:7s} {'ok' if not row['failures'] else 'FAIL'}")

    # ---------------- aggregate ----------------
    ans = [r for r in results if r["category"] != "unsupported"]
    uns = [r for r in results if r["category"] == "unsupported"]
    cross = [r for r in results if r["category"] == "cross_doc"]
    metrics = {
        "n_cases": len(results),
        "retrieval_hit_rate_pct": pct(sum(r.get("retrieval_hit", False) for r in ans), len(ans)),
        "page_hit_rate_pct": pct(sum(r.get("page_hit", False) for r in ans), len(ans)),
        "cross_doc_hit_rate_pct": pct(sum(r.get("retrieval_hit", False) for r in cross), len(cross)),
        "mrr": round(statistics.mean(1 / r["first_rank"] if r.get("first_rank") else 0 for r in ans), 3) if ans else None,
        "wrong_abstention_pct": pct(sum(r.get("wrongly_abstained", False) for r in ans), len(ans)),
        "unsupported_abstain_pct": pct(sum(r.get("abstain_correct", False) for r in uns), len(uns)),
    }
    if llm != "none":
        graded = [r for r in ans if "answer_correct" in r]
        metrics["answer_accuracy_pct"] = pct(sum(r["answer_correct"] for r in graded), len(ans))
        metrics["citation_correct_pct"] = pct(sum(r.get("citation_correct", False) for r in graded), len(ans))
    rl = [r["retrieval_ms"] for r in results if "retrieval_ms" in r]
    tl = [r["total_ms"] for r in results if "total_ms" in r]
    metrics.update({
        "retrieval_ms_p50": percentile(rl, 50), "retrieval_ms_p95": percentile(rl, 95),
        "total_ms_p50": percentile(tl, 50), "total_ms_p95": percentile(tl, 95),
    })

    sweep = []
    a_scores = [r["top_score"] for r in ans if "top_score" in r]
    u_scores = [r["top_score"] for r in uns if "top_score" in r]
    for thr in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        sweep.append({
            "threshold": thr,
            "answerable_wrongly_blocked_pct": pct(sum(s < thr for s in a_scores), len(a_scores)),
            "unsupported_blocked_pct": pct(sum(s < thr for s in u_scores), len(u_scores)),
        })

    out = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "label": args.label,
        "config": {"llm": llm, "model": config.GROQ_MODEL if llm == "groq" else llm, "k": k,
                   "min_score": min_score, "chunk_size": config.CHUNK_SIZE,
                   "chunk_overlap": config.CHUNK_OVERLAP},
        "metrics": metrics, "threshold_sweep": sweep, "cases": results,
    }
    results_dir = HERE / "results"
    results_dir.mkdir(exist_ok=True)
    previous = sorted(results_dir.glob("*.json"))
    prev = json.loads(previous[-1].read_text(encoding="utf-8")) if previous else None
    fname = results_dir / f"{datetime.datetime.now():%Y%m%d_%H%M%S}_{args.label}.json"
    fname.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # ---------------- report ----------------
    print("\n=== RESULTS (" + f"llm={llm}, k={k}, min_score={min_score}" + ") ===")
    for key, val in metrics.items():
        line = f"  {key:28s} {val}"
        if prev and prev["metrics"].get(key) is not None and val is not None:
            d = round(val - prev["metrics"][key], 3)
            line += f"   (vs previous: {'+' if d >= 0 else ''}{d})"
        print(line)
    fails = [r for r in results if r["failures"]]
    print(f"\n=== FAILURES ({len(fails)}) ===")
    for r in fails:
        print(f"  {r['id']}: {r['question']}")
        for f in r["failures"]:
            print(f"      -> {f}")
    print("\n=== THRESHOLD SWEEP (pick the lowest threshold that blocks unsupported questions) ===")
    print("  threshold | answerable wrongly blocked | unsupported blocked")
    for s in sweep:
        print(f"    {s['threshold']:.2f}    |        {str(s['answerable_wrongly_blocked_pct']):>6}%          |     {str(s['unsupported_blocked_pct']):>6}%")
    print(f"\nSaved: {fname.relative_to(ROOT)}")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()