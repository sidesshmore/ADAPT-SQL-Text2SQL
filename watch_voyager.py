"""
Live EX watcher for Voyager eval results.

Usage (local):
    python watch_voyager.py

Usage (SOL, auto-refresh):
    watch -n 30 'cd /scratch/smore123/ADAPT-SQL-Text2SQL && python watch_voyager.py --sol'

Reads checkpoint .pkl files (required format for merge_checkpoints.py compatibility).
"""
import argparse
import glob
import os
import pickle  # required: matches checkpoint format used across the pipeline
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))


def load_latest_checkpoint(files: list) -> list:
    latest = max(files, key=os.path.getmtime)
    try:
        data = pickle.load(open(latest, "rb"))
        return data.get("results", [])
    except Exception:
        return []


def get_ex(result: dict) -> bool:
    return result.get("result", {}).get("step11", {}).get("execution_accuracy", False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="eval_results_voyager",
                        help="Root dir containing range_*/ subdirectories")
    parser.add_argument("--sol", action="store_true",
                        help="Use SOL path: /scratch/smore123/eval_results_voyager")
    args = parser.parse_args()

    base = Path("/scratch/smore123/eval_results_voyager") if args.sol else PROJECT_DIR / args.dir

    range_files: dict = defaultdict(list)
    for f in glob.glob(str(base / "range_*/checkpoint_*.pkl")) + \
             glob.glob(str(base / "range_*/final_checkpoint_*.pkl")):
        range_files[Path(f).parent.name].append(f)

    if not range_files:
        print(f"No checkpoints found in {base}/range_*/")
        print("Waiting for first checkpoint (~25 queries in)...")
        return

    total, correct = 0, 0
    rows = []

    for rng in sorted(range_files):
        results = load_latest_checkpoint(range_files[rng])
        if not results:
            continue
        c = sum(get_ex(r) for r in results)
        n = len(results)
        pct = c / n * 100 if n else 0
        is_final = any("final_checkpoint" in f for f in range_files[rng])
        rows.append((rng, c, n, pct, "DONE" if is_final else "running"))
        total += n
        correct += c

    if not rows:
        print("No results yet.")
        return

    print(f"\n{'Range':<20} {'Correct':>8} {'Total':>6} {'EX':>7}  Status")
    print("-" * 55)
    for rng, c, n, pct, status in rows:
        print(f"{rng:<20} {c:>8} {n:>6} {pct:>6.1f}%  {status}")
    print("-" * 55)
    overall = correct / total * 100 if total else 0
    print(f"{'OVERALL':<20} {correct:>8} {total:>6} {overall:>6.1f}%\n")


if __name__ == "__main__":
    main()
