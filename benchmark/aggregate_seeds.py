#!/usr/bin/env python3
import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

_BENCH_ROOT = Path(__file__).resolve().parent.parent
if str(_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCH_ROOT))
from benchmark.benchmark_utils import fmt_sig


NUMERIC_COLS = [
    "Params(M)",
    "FLOPs(G)",
    "Train(h)",
    "Infer(ms)",
    "MSE",
    "MAE",
    "PSNR",
    "SSIM",
    "L1-Grad",
    "LPIPS",
]


def parse_args():
    p = argparse.ArgumentParser("Aggregate benchmark csv over seeds")
    p.add_argument("--in-csv", required=True)
    p.add_argument("--out-csv", required=True)
    return p.parse_args()


def to_float(x):
    try:
        if x is None or x == "" or str(x).upper() == "NA":
            return None
        return float(x)
    except Exception:
        return None


def fmt(mu, sd):
    """至少 4 位有效数字，便于区分模型差异"""
    if mu is None:
        return "NA"
    if sd is None:
        return fmt_sig(mu)
    return f"{fmt_sig(mu)}±{fmt_sig(sd)}"


def main():
    args = parse_args()
    groups = defaultdict(list)

    with open(args.in_csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            # allow "Model#seed42" naming
            model = row["Model"].split("#seed")[0]
            key = (model, row.get("Model Category", "NA"))
            groups[key].append(row)

    out_rows = []
    for (model, category), rows in groups.items():
        agg = {"Model": model, "Model Category": category, "Runs": str(len(rows))}
        for c in NUMERIC_COLS:
            vals = [to_float(r.get(c, "")) for r in rows]
            vals = [v for v in vals if v is not None and math.isfinite(v)]
            if len(vals) == 0:
                agg[c] = "NA"
            elif len(vals) == 1:
                agg[c] = fmt_sig(vals[0])
            else:
                agg[c] = fmt(mean(vals), pstdev(vals))
        out_rows.append(agg)

    fieldnames = ["Model", "Model Category", "Runs"] + NUMERIC_COLS
    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in out_rows:
            w.writerow(row)

    print(f"Saved: {args.out_csv}")


if __name__ == "__main__":
    main()

