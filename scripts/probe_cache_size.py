"""
Estimate full-scale fit_target_index.py disk usage from a small sample,
before running the real multi-hour fit. Safe to run repeatedly — writes
only to a throwaway output directory, nothing under version control or
the real cache/ tree is touched. Run with CWD at the repo root.

Usage:
    python scripts/probe_cache_size.py \
        --source dataset/train/train_source3.tsv \
        --sample-size 200000 \
        --full-scale-size 5000000 \
        --n-combos 4 \
        --out-dir /tmp/disk_probe
"""
import argparse
import random
import shutil
import sys
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                       / "code" / "business_entity_resolution" / "src"))

from blocking.fit_target_index import fit_target_index


def dir_size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="a real source TSV to sample rows from")
    p.add_argument("--sample-size", type=int, default=200_000)
    p.add_argument("--full-scale-size", type=int, default=5_000_000)
    p.add_argument("--n-combos", type=int, default=4,
                   help="split x target_source combinations to extrapolate across")
    p.add_argument("--out-dir", default="/tmp/disk_probe")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "fitted").mkdir(parents=True)

    # Single pass: count lines, then parse a uniform random subset only.
    with open(args.source, encoding="utf-8") as f:
        total = sum(1 for _ in f) - 1
    keep = set(random.Random(0).sample(range(total), min(args.sample_size, total)))

    import pandas as pd
    rows = []
    with open(args.source, encoding="utf-8") as f:
        header = f.readline()
        for i, line in enumerate(f):
            if i in keep:
                rows.append(line)
    sample = pd.read_csv(
        StringIO(header + "".join(rows)), sep="\t", dtype=str,
    ).fillna("")
    assert {"entity_id", "business_name", "business_address"} <= set(sample.columns)

    fit_target_index(sample, str(out_dir / "fitted" / "probe"))

    sample_bytes = dir_size_bytes(out_dir / "fitted")
    scale = args.full_scale_size / len(sample)
    est_per_combo = sample_bytes * scale
    est_total = est_per_combo * args.n_combos
    free_bytes = shutil.disk_usage(".").free

    def gb(b):
        return b / (1024 ** 3)

    print(f"Sample rows: {len(sample):,}  |  Sample artifacts: {gb(sample_bytes):.3f} GB")
    print(f"Linear extrapolation to {args.full_scale_size:,}/combo: {gb(est_per_combo):.2f} GB")
    print(f"Across {args.n_combos} combos: {gb(est_total):.2f} GB estimated total")
    print(f"Free disk right now: {gb(free_bytes):.2f} GB")
    if est_total > free_bytes:
        print("WARNING: estimated cache exceeds free disk.")
        print("Fallback order: (1) tighten min_df/max_df; (2) fit+use+discard "
              "one combo at a time; (3) external disk.")
    else:
        print("OK: estimated cache fits (linear extrapolation is a conservative "
              "upper bound — vocab growth is sublinear under min_df/max_df).")


if __name__ == "__main__":
    main()
