"""
Merge final checkpoint files from parallel batch runs into one CSV.

The checkpoint format (pickle) is the same format used by batch_utils.save_checkpoint —
these are our own files, not external input.

Usage:
    python merge_checkpoints.py \
        batch_results/range_0_200/final_checkpoint_*.pkl \
        batch_results/range_200_400/final_checkpoint_*.pkl \
        batch_results/range_400_600/final_checkpoint_*.pkl \
        batch_results/range_600_800/final_checkpoint_*.pkl \
        --out merged_results

Or with a glob pattern:
    python merge_checkpoints.py \
        --auto "batch_results/range_*/final_checkpoint_*.pkl" \
        --out merged_results
"""
import argparse
import glob
import pickle  # same format used by batch_utils.save_checkpoint
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from ui.batch_utils import generate_comprehensive_csv


def load_pkl(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", help="Checkpoint .pkl files to merge")
    parser.add_argument("--auto", help="Glob pattern to auto-discover files (quote it)")
    parser.add_argument("--out", default="./merged_results", help="Output directory for merged CSV")
    args = parser.parse_args()

    paths = list(args.files)
    if args.auto:
        paths += sorted(glob.glob(args.auto))

    if not paths:
        print("No checkpoint files specified. Use positional args or --auto <glob>.")
        sys.exit(1)

    all_results = []
    for p in paths:
        data = load_pkl(p)
        all_results.extend(data["results"])
        print(f"  Loaded {data['num_results']} results from {p}")

    # Sort by original index so CSV row order matches Spider dev.json order
    all_results.sort(key=lambda r: r["index"])

    # Deduplicate (in case of overlap)
    seen = set()
    deduped = []
    for r in all_results:
        if r["index"] not in seen:
            seen.add(r["index"])
            deduped.append(r)

    if len(deduped) < len(all_results):
        print(f"  Deduplicated: {len(all_results)} → {len(deduped)} results")
    all_results = deduped

    print(f"\nTotal merged: {len(all_results)} results")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out_dir = Path(args.out)
    csv_path = generate_comprehensive_csv(all_results, timestamp, out_dir)
    print(f"CSV saved to: {csv_path}")

    # Save a merged checkpoint for potential further use
    merged_pkl = out_dir / f"merged_final_{timestamp.replace(' ', '_').replace(':', '-')}.pkl"
    with open(merged_pkl, "wb") as f:
        pickle.dump({
            "results": all_results,
            "timestamp": timestamp,
            "num_results": len(all_results),
            "saved_at": timestamp,
            "is_final": True,
        }, f)
    print(f"Merged checkpoint saved to: {merged_pkl}")


if __name__ == "__main__":
    main()
