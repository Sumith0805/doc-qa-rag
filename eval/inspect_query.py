"""Debug tool: show exactly what retrieval returns for ONE question.

    python -m eval.inspect_query "your question" --expect "h = 8"

Indexes the eval documents into a throw-away folder, then prints, for the question and for each
sub-question it gets split into: the dense-only ranking, the hybrid ranking, and the final
merged result. `--expect` marks chunks that contain the text you are hoping to find.
"""
import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


def show(title, hits, expect):
    print(f"--- {title}")
    for rank, h in enumerate(hits, start=1):
        mark = "  <== EXPECTED TEXT" if expect and expect.lower() in h["text"].lower() else ""
        snippet = " ".join(h["text"].split())[:70].encode("ascii", "replace").decode()
        print(f"  {rank}. {h['filename'][:12]:12s} p{h['page']:<3d} cos={h['score']:.3f} "
              f"bm25={h.get('bm25', '-')!s:>6}  {snippet}{mark}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--expect", default=None, help="text you expect inside a retrieved chunk")
    args = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="documind_inspect_")
    os.environ["DOCUMIND_DATA_DIR"] = tmp  # before importing config
    import config
    import hybrid
    import store

    for path in sorted((HERE / "docs").iterdir()):
        store.index_document(str(path), path.name)

    parts = hybrid.split_question(args.question)
    print(f"\nSub-questions: {parts}\n")
    for q in parts:
        print(f"=== {q}")
        config.HYBRID = False
        show("dense only", store.search(q, args.k), args.expect)
        config.HYBRID = True
        show("hybrid (dense + BM25)", store.search(q, args.k), args.expect)
        print()
    print("=== FINAL result for the full question (what the LLM would see)")
    show("merged", store.search(args.question, args.k), args.expect)
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()