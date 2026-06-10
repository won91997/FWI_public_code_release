#!/usr/bin/env python3
"""Refresh all summary tables/CSVs after SSIM fix (read metrics.json)."""
from __future__ import print_function

import csv
import json
import os
from pathlib import Path

RC = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("RESULT_ROOT", str(RC)))
SEEDS = ["42"]

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

TEN_MODELS = ["ConvNeXtKaggle"] + [x[0] for x in MAIN8] + ["VIFNet"]


def fair_eval_dir(model, seed):
    seed = str(seed)
    if model == "ConvNeXtKaggle":
        return DATA / "benchmark_logs_convnext_kaggle_seed42_b2" / (
            "convnext_kaggle_fair_seed{}".format(seed)
        ) / "eval"
    if model == "VIFNet":
        return DATA / "benchmark_logs_fair_nativeloss_vifnet_seed42" / (
            "vifnet_seed{}".format(seed)
        ) / "eval"
    for name, prefix, sub in MAIN8:
        if name == model:
            return DATA / "benchmark_logs_fair_nativeloss_seed42_all" / (
                "{}_seed{}".format(prefix, seed)
            ) / sub
    raise KeyError(model)


def nonfair_eval_dir(model):
    if model == "ConvNeXtKaggle":
        return RC / "logs_nonfair/convnext_kaggle_nonfair_seed42/eval"
    if model == "VIFNet":
        return RC / "logs_nonfair/vifnet_nonfair_seed42/eval"
    for name, prefix, sub in MAIN8:
        if name == model:
            return RC / "logs_nonfair/main_10models_nonfair/{}_seed42".format(prefix) / sub
    raise KeyError(model)


def load_metrics(evdir):
    p = Path(evdir) / "metrics.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def patch_csv_ssim(csv_path, key_fn):
    """key_fn(row) -> (model_base, seed) or None to skip."""
    if not csv_path.is_file():
        return
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    if not rows:
        return
    n = 0
    for row in rows:
        ks = key_fn(row)
        if not ks:
            continue
        model, seed = ks
        try:
            ev = fair_eval_dir(model, seed) if seed else nonfair_eval_dir(model)
        except KeyError:
            continue
        m = load_metrics(ev)
        if m and "SSIM" in m:
            row["SSIM"] = str(m["SSIM"])
            n += 1
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print("patched SSIM in {} ({} rows)".format(csv_path, n))


def key_hash_model(row):
    raw = row.get("Model", "")
    if "#seed" not in raw:
        return None
    base, seed = raw.split("#seed", 1)
    return base, seed


def update_rc_seed42_csv(src_csv, mode_label):
    """Rewrite SSIM column from metrics.json; keep other columns."""
    rows = list(csv.DictReader(src_csv.open(encoding="utf-8")))
    for row in rows:
        model = row["Model"]
        m = load_metrics(fair_eval_dir(model, 42))
        if m:
            row["SSIM"] = str(m["SSIM"])
    with src_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print("updated", src_csv)


def build_fair_seed42agg_csv(out_path):
    fields = [
        "Model", "Seed", "MSE", "MAE", "PSNR", "SSIM", "L1-Grad", "LPIPS", "Status", "EvalDir"
    ]
    rows = []
    for seed in SEEDS:
        for model in TEN_MODELS:
            ev = fair_eval_dir(model, seed)
            m = load_metrics(ev)
            if not m:
                rows.append(
                    {
                        "Model": model,
                        "Seed": seed,
                        "Status": "missing",
                        "EvalDir": str(ev),
                    }
                )
                continue
            rows.append(
                {
                    "Model": model,
                    "Seed": seed,
                    "MSE": m.get("MSE"),
                    "MAE": m.get("MAE"),
                    "PSNR": m.get("PSNR"),
                    "SSIM": m.get("SSIM"),
                    "L1-Grad": m.get("L1-Grad"),
                    "LPIPS": m.get("LPIPS"),
                    "Status": "OK",
                    "EvalDir": str(ev),
                }
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("wrote", out_path)


def build_nonfair_md_csv():
    """Nonfair 10-model seed42 summary from RC logs_nonfair + plot csv efficiency cols."""
    plot_csv = RC / "logs_nonfair/nonfair_10models_metrics.csv"
    eff = {}
    if plot_csv.is_file():
        for row in csv.DictReader(plot_csv.open(encoding="utf-8")):
            name = row["model"].replace("TU-Net", "TU_Net").replace("ABA-FWI", "ABA_FWI")
            eff[name] = row

    rows_md = []
    for model in TEN_MODELS:
        m = load_metrics(nonfair_eval_dir(model))
        e = eff.get(model, {})
        if not m:
            continue
        rows_md.append(
            {
                "model": model,
                "Params_M": e.get("Params_M", ""),
                "FLOPs_G": e.get("FLOPs_G", ""),
                "Train_h": e.get("Train_h", ""),
                "Infer_ms": e.get("Infer_ms", ""),
                "MSE": m.get("MSE"),
                "MAE": m.get("MAE"),
                "PSNR": m.get("PSNR"),
                "SSIM": m.get("SSIM"),
                "L1-Grad": m.get("L1-Grad"),
                "LPIPS": m.get("LPIPS"),
            }
        )

    csv_out = RC / "results/resolution_constrained_nonfair_seed42_b2_10models.csv"
    md_out = RC / "results/resolution_constrained_nonfair_seed42_b2_10models.md"

    fields = [
        "model", "Params_M", "FLOPs_G", "Train_h", "Infer_ms",
        "MSE", "MAE", "PSNR", "SSIM", "L1-Grad", "LPIPS",
    ]
    with csv_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_md)
    print("wrote", csv_out)

    lines = [
        "# Resolution-Constrained 10 模型汇总（Nonfair / Full-Resolution）",
        "",
        "配置：`nonfair` + `seed42` + `batch=2` + test 集指标",
        "",
        "SSIM 已按 `data_range = max(GT)-min(GT)`（逐样本）重算。",
        "",
        "## 总表",
        "",
        "| 模型 | Params(M) | FLOPs(G) | Train(h) | Infer(ms) | MSE | MAE | PSNR | SSIM | L1-Grad | LPIPS |",
        "|------|----------:|---------:|---------:|----------:|----:|----:|-----:|-----:|--------:|------:|",
    ]
    for r in sorted(rows_md, key=lambda x: -(float(x["PSNR"]) if x["PSNR"] else 0)):
        lines.append(
            "| {model} | {Params_M} | {FLOPs_G} | {Train_h} | {Infer_ms} | "
            "{MSE} | {MAE} | {PSNR} | {SSIM} | {L1-Grad} | {LPIPS} |".format(**r)
        )
    lines += ["", "## 按 PSNR 排名", "", "| 排名 | 模型 | PSNR | SSIM | LPIPS |", "|------|------|-----:|-----:|------:|"]
    for i, r in enumerate(sorted(rows_md, key=lambda x: -(float(x["PSNR"]) if x["PSNR"] else 0)), 1):
        lines.append("| {} | {} | {} | {} | {} |".format(i, r["model"], r["PSNR"], r["SSIM"], r["LPIPS"]))
    md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", md_out)


def rerun_wyfwyf_nonfair_ssim():
    """Patch wyfwyf nonfair metrics.json SSIM where pred exists."""
    import sys
    sys.path.insert(0, str(RC.parent))
    from benchmark.benchmark_eval import ssim_global, squeeze_hw
    from benchmark.benchmark_utils import fmt_sig
    import numpy as np
    import shutil

    mapping = {
        "aba_fwi": "aba_fwi",
        "convnext_kaggle": "ConvNeXtKaggle",
        "dcnet": "DCNet",
        "ddnet70": "DDNet70",
        "inversionnet": "InversionNet",
        "tu_net": "TU_Net",
        "vifnet": "VIFNet",
        "futefwi": "FuteFWI",
        "velocitygan": "VelocityGAN",
    }
    base = DATA / "benchmark_logs_nonfair_nativeloss_seed42"
    for folder in sorted(base.iterdir()):
        if not folder.is_dir() or "_seed" not in folder.name:
            continue
        for sub in ("eval", "eval_results"):
            ev = folder / sub
            pred, gt, mj = ev / "pred.npy", ev / "gt.npy", ev / "metrics.json"
            if not (pred.is_file() and gt.is_file() and mj.is_file()):
                continue
            pred_a = squeeze_hw(np.load(str(pred)))
            gt_a = squeeze_hw(np.load(str(gt)))
            metrics = json.loads(mj.read_text())
            bak = ev / "metrics.json.bak_ssim"
            if not bak.is_file():
                shutil.copy2(str(mj), str(bak))
            metrics["SSIM"] = float(fmt_sig(ssim_global(pred_a, gt_a)))
            mj.write_text(json.dumps(metrics, indent=2) + "\n")
            print("wyfwyf nonfair SSIM", folder.name, metrics["SSIM"])


def main():
    res = RC / "results"
    update_rc_seed42_csv(res / "resolution_constrained_fair_seed42_b2_10models.csv", "fair")
    update_rc_seed42_csv(res / "resolution_constrained_target_resolution_seed42_b2_10models.csv", "Target-Resolution")
    build_fair_seed42agg_csv(res / "resolution_constrained_fair_seed42agg_10models.csv")
    build_nonfair_md_csv()
    rerun_wyfwyf_nonfair_ssim()
    patch_csv_ssim(DATA / "benchmark_logs_fair_nativeloss_seed42_all/benchmark_metrics.csv", key_hash_model)
    patch_csv_ssim(DATA / "benchmark_logs_fair_nativeloss_seed42_all/benchmark_metrics_agg.csv", key_hash_model)
    patch_csv_ssim(DATA / "benchmark_logs_nonfair_nativeloss_seed42/benchmark_metrics.csv", key_hash_model)
    patch_csv_ssim(DATA / "benchmark_logs_nonfair_nativeloss_seed42/benchmark_metrics_agg.csv", key_hash_model)


if __name__ == "__main__":
    main()
