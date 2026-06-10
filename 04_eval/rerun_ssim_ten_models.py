#!/usr/bin/env python3
"""Recompute SSIM only for 10-model fair/nonfair test eval."""
from __future__ import print_function

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

BENCH = Path(__file__).resolve().parent
RC = BENCH.parent
ROOT = RC.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.benchmark_eval import ssim_global, squeeze_hw
from benchmark.benchmark_utils import fmt_sig

DATA = Path(os.environ.get("RESULT_ROOT", str(RC)))
FAIR_MAIN = DATA / "benchmark_logs_fair_nativeloss_seed42_all"
FAIR_CNX_ROOT = DATA / "benchmark_logs_convnext_kaggle_seed42_b2"
FAIR_VIF_ROOT = DATA / "benchmark_logs_fair_nativeloss_vifnet_seed42"
NONFAIR_MAIN = RC / "logs_nonfair" / "main_10models_nonfair"
NONFAIR_CNX = RC / "logs_nonfair" / "convnext_kaggle_nonfair_seed42" / "eval"
NONFAIR_VIF = RC / "logs_nonfair" / "vifnet_nonfair_seed42" / "eval"

MAIN8 = [
    ("FuteFWI", "futefwi", "eval_results"),
    ("InversionNet", "inversionnet", "eval"),
    ("VelocityGAN", "velocitygan", "eval_results"),
    ("DCNet", "dcnet", "eval"),
    ("DDNet70", "ddnet70", "eval"),
    ("TU_Net", "tu_net", "eval"),
    ("ABA_FWI", "aba_fwi", "eval"),
    ("FCNVMB", "fcnvmb", "eval"),
]


def _fair_models(seed):
    models = [
        (
            "ConvNeXtKaggle",
            FAIR_CNX_ROOT / "convnext_kaggle_fair_seed{}".format(seed) / "eval",
        )
    ]
    for name, prefix, sub in MAIN8:
        models.append((name, FAIR_MAIN / "{}_seed{}".format(prefix, seed) / sub))
    models.append(("VIFNet", FAIR_VIF_ROOT / "vifnet_seed{}".format(seed) / "eval"))
    return models


def _nonfair_models(seed=42):
    if seed != 42:
        return []
    return [
        ("ConvNeXtKaggle", NONFAIR_CNX),
    ] + [
        (name, NONFAIR_MAIN / "{}_seed42".format(prefix) / sub)
        for name, prefix, sub in MAIN8
    ] + [("VIFNet", NONFAIR_VIF)]


def _patch_ssim(name, evdir):
    evdir = Path(evdir)
    pred_p = evdir / "pred.npy"
    gt_p = evdir / "gt.npy"
    out_p = evdir / "metrics.json"
    if not pred_p.is_file() or not gt_p.is_file():
        return {"model": name, "status": "skip", "reason": "missing pred/gt", "path": str(evdir)}
    if not out_p.is_file():
        return {"model": name, "status": "skip", "reason": "missing metrics.json", "path": str(evdir)}

    pred = squeeze_hw(np.load(str(pred_p)))
    gt = squeeze_hw(np.load(str(gt_p)))
    if pred.shape != gt.shape:
        return {
            "model": name,
            "status": "error",
            "reason": "shape mismatch",
            "path": str(evdir),
        }

    bak = evdir / "metrics.json.bak_ssim"
    if not bak.is_file():
        shutil.copy2(str(out_p), str(bak))

    metrics = json.loads(out_p.read_text(encoding="utf-8"))
    old_ssim = metrics.get("SSIM")
    new_ssim = float(fmt_sig(ssim_global(pred, gt)))
    metrics["SSIM"] = new_ssim
    out_p.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return {
        "model": name,
        "status": "ok",
        "path": str(evdir),
        "SSIM_old": old_ssim,
        "SSIM_new": new_ssim,
    }


def _run_group(label, models):
    rows = []
    print("\n=== {} ===".format(label))
    for name, evdir in models:
        r = _patch_ssim(name, evdir)
        rows.append(r)
        if r["status"] == "ok":
            print(
                "{:16s} SSIM {} -> {}  {}".format(
                    name, r["SSIM_old"], r["SSIM_new"], evdir
                )
            )
        else:
            print("{:16s} {}  {}  ({})".format(name, r["status"], r.get("reason", ""), evdir))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fair-seeds",
        default="42",
        help="comma-separated seeds for fair 10-model rerun",
    )
    ap.add_argument("--skip-nonfair", action="store_true")
    args = ap.parse_args()

    all_rows = {"fair": {}, "nonfair": {}}
    for seed in [s.strip() for s in args.fair_seeds.split(",") if s.strip()]:
        rows = _run_group("FAIR seed{} test".format(seed), _fair_models(seed))
        all_rows["fair"][seed] = rows

    if not args.skip_nonfair:
        all_rows["nonfair"]["42"] = _run_group("NONFAIR seed42 test", _nonfair_models(42))

    summary = RC / "results" / "ssim_rerun_fair_nonfair.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(all_rows, indent=2) + "\n", encoding="utf-8")
    print("\nSaved:", summary)


if __name__ == "__main__":
    main()
