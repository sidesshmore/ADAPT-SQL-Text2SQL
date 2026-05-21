"""
Live EX watcher for Voyager eval results.

Supports two layouts:
  Single model:  base/range_*/
  Multi-model:   base/<model>/<split>/range_*/

Usage (local):
    python watch_voyager.py
    python watch_voyager.py --sol

Usage (SOL, auto-refresh):
    watch -n 60 'cd /scratch/smore123/ADAPT-SQL-Text2SQL && source venv/bin/activate && python3 watch_voyager.py --sol'
"""
import argparse
import glob
import os
import pickle  # matches checkpoint format used across the pipeline
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


def parse_range_size(range_name: str) -> int:
    """Parse expected query count from dir name like range_0_150 → 150."""
    parts = range_name.split("_")
    if len(parts) == 3:
        try:
            return int(parts[2]) - int(parts[1])
        except ValueError:
            pass
    return 0


def print_split_table(split_dir: Path, label: str) -> bool:
    """Print EX table for one model+split directory containing range_*/ subdirs."""
    range_files: dict = defaultdict(list)
    for f in glob.glob(str(split_dir / "range_*/checkpoint_*.pkl")) + \
             glob.glob(str(split_dir / "range_*/final_checkpoint_*.pkl")):
        range_files[Path(f).parent.name].append(f)

    if not range_files:
        return False

    total_done, total_expected, correct = 0, 0, 0
    rows = []
    for rng in sorted(range_files):
        results = load_latest_checkpoint(range_files[rng])
        if not results:
            continue
        c = sum(get_ex(r) for r in results)
        n = len(results)
        expected = parse_range_size(rng) or n
        pct = c / n * 100 if n else 0
        is_final = any("final_checkpoint" in f for f in range_files[rng])
        status = "DONE" if is_final else f"{n}/{expected}"
        rows.append((rng, c, n, expected, pct, status, is_final))
        total_done += n
        total_expected += expected
        correct += c

    if not rows:
        return False

    overall = correct / total_done * 100 if total_done else 0
    done_ranges = sum(1 for *_, is_final in rows if is_final)
    num_ranges = len(rows)
    print(f"\n{'='*68}")
    print(f"  {label}")
    print(f"  Queries: {total_done}/{total_expected}  EX: {correct}/{total_done} = {overall:.1f}%  ({done_ranges}/{num_ranges} ranges done)")
    print(f"{'='*68}")
    print(f"  {'Range':<22} {'Correct':>7} {'Done/Exp':>10} {'EX%':>7}  Status")
    print(f"  {'-'*60}")
    for rng, c, n, expected, pct, status, _ in rows:
        done_exp = f"{n}/{expected}"
        print(f"  {rng:<22} {c:>7} {done_exp:>10} {pct:>6.1f}%  {status}")
    print(f"  {'-'*60}")
    overall_done_exp = f"{total_done}/{total_expected}"
    print(f"  {'OVERALL':<22} {correct:>7} {overall_done_exp:>10} {overall:>6.1f}%")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="eval_results_voyager",
                        help="Root results directory")
    parser.add_argument("--sol", action="store_true",
                        help="Use SOL path: /scratch/smore123/eval_results_voyager")
    args = parser.parse_args()

    base = Path("/scratch/smore123/eval_results_voyager") if args.sol else PROJECT_DIR / args.dir

    if not base.exists():
        print(f"No results directory found at {base}")
        return

    found_any = False

    subdirs = [d for d in sorted(base.iterdir()) if d.is_dir()]
    if not subdirs:
        print(f"No results found in {base}")
        return

    # Always show flat range_*/ entries under base as the legacy job
    if any(d.name.startswith("range_") for d in subdirs):
        found_any |= print_split_table(base, "qwen3-coder-30b [dev] (legacy job)")

    # Also show all model subdirs (new multi-model layout)
    for model_dir in sorted(d for d in subdirs if not d.name.startswith("range_")):
        model_label = model_dir.name.replace("___", "/")
        split_dirs = [d for d in sorted(model_dir.iterdir()) if d.is_dir()] if model_dir.is_dir() else []
        if not split_dirs:
            continue
        if any(d.name.startswith("range_") for d in split_dirs):
            found_any |= print_split_table(model_dir, model_label)
        else:
            for split_dir in split_dirs:
                label = f"{model_label}  [{split_dir.name}]"
                found_any |= print_split_table(split_dir, label)

    if not found_any:
        print(f"No checkpoints found in {base}")
        print("Waiting for first checkpoint (~25 queries in)...")


if __name__ == "__main__":
    main()
