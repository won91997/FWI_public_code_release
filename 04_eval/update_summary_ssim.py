#!/usr/bin/env python3
"""Refresh SSIM column in summary md/csv from updated metrics.json."""
import csv
import json
import os
import re
from pathlib import Path

RC = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("RESULT_ROOT", str(RC)))

FAIR_PATHS = {
    "ConvNeXtKaggle": DATA / "benchmark_logs_convnext_kaggle_seed42_b2/convnext_kaggle_fair_seed42/eval/metrics.json",
    "FuteFWI": DATA / "benchmark_logs_fair_nativeloss_seed42_all/futefwi_seed42/eval_results/metrics.json",
    "InversionNet": DATA / "benchmark_logs_fair_nativeloss_seed42_all/inversionnet_seed42/eval/metrics.json",
    "VelocityGAN": DATA / "benchmark_logs_fair_nativeloss_seed42_all/velocitygan_seed42/eval_results/metrics.json",
    "DCNet": DATA / "benchmark_logs_fair_nativeloss_seed42_all/dcnet_seed42/eval/metrics.json",
    "DDNet70": DATA / "benchmark_logs_fair_nativeloss_seed42_all/ddnet70_seed42/eval/metrics.json",
    "TU_Net": DATA / "benchmark_logs_fair_nativeloss_seed42_all/tu_net_seed42/eval/metrics.json",
    "ABA_FWI": DATA / "benchmark_logs_fair_nativeloss_seed42_all/aba_fwi_seed42/eval/metrics.json",
    "FCNVMB": DATA / "benchmark_logs_fair_nativeloss_seed42_all/fcnvmb_seed42/eval/metrics.json",
    "VIFNet": DATA / "benchmark_logs_fair_nativeloss_vifnet_seed42/vifnet_seed42/eval/metrics.json",
}

NONFAIR_PATHS = {
    "ConvNeXtKaggle": RC / "logs_nonfair/convnext_kaggle_nonfair_seed42/eval/metrics.json",
    "FuteFWI": RC / "logs_nonfair/main_10models_nonfair/futefwi_seed42/eval_results/metrics.json",
    "InversionNet": RC / "logs_nonfair/main_10models_nonfair/inversionnet_seed42/eval/metrics.json",
    "VelocityGAN": RC / "logs_nonfair/main_10models_nonfair/velocitygan_seed42/eval_results/metrics.json",
    "DCNet": RC / "logs_nonfair/main_10models_nonfair/dcnet_seed42/eval/metrics.json",
    "DDNet70": RC / "logs_nonfair/main_10models_nonfair/ddnet70_seed42/eval/metrics.json",
    "TU_Net": RC / "logs_nonfair/main_10models_nonfair/tu_net_seed42/eval/metrics.json",
    "ABA_FWI": RC / "logs_nonfair/main_10models_nonfair/aba_fwi_seed42/eval/metrics.json",
    "FCNVMB": RC / "logs_nonfair/main_10models_nonfair/fcnvmb_seed42/eval/metrics.json",
    "VIFNet": RC / "logs_nonfair/vifnet_nonfair_seed42/eval/metrics.json",
}

# CSV 里 TU-Net / ABA-FWI 带连字符
NONFAIR_CSV_ALIASES = {"TU_Net": "TU-Net", "ABA_FWI": "ABA-FWI", "FCNVMB": "FCNVMB"}


def _load_ssim(mapping):
    out = {}
    for model, p in mapping.items():
        if p.is_file():
            out[model] = json.loads(p.read_text())["SSIM"]
    return out


def _patch_md_table_line(line, ssim_map):
    stripped = line.strip()
    if not stripped.startswith("|"):
        return line
    parts = [x.strip() for x in stripped.split("|")]
    inner = parts[1:-1]
    if not inner:
        return line
    model = inner[0]
    # 总表: | model | Params | ... | PSNR | SSIM | L1-Grad | ...
    if model in ssim_map and len(inner) >= 9 and inner[0] in ssim_map:
        # PSNR 在第 8 列(0-based index 7), SSIM 在 index 8
        if re.match(r"^[\d.]+$", inner[7].replace(".", "", 1)) and "Grad" not in inner[8]:
            inner[8] = str(ssim_map[model])
            return "| " + " | ".join(inner) + " |\n"
    # 排名表: | rank | model | PSNR | SSIM | LPIPS |
    if len(inner) >= 5 and inner[1] in ssim_map and inner[0].isdigit():
        inner[3] = str(ssim_map[inner[1]])
        return "| " + " | ".join(inner) + " |\n"
    return line


def patch_md(md_path, ssim_map):
    text = md_path.read_text(encoding="utf-8")
    lines = []
    for line in text.splitlines(keepends=True):
        lines.append(_patch_md_table_line(line, ssim_map))
    md_path.write_text("".join(lines), encoding="utf-8")
    print("updated md:", md_path)


def patch_nonfair_csv(csv_path, ssim_map):
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    for row in rows:
        name = row["model"]
        for model, ssim in ssim_map.items():
            alias = NONFAIR_CSV_ALIASES.get(model, model)
            if name == alias or name == model:
                row["SSIM"] = str(ssim)
    fields = rows[0].keys() if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("updated csv:", csv_path)


def main():
    fair = _load_ssim(FAIR_PATHS)
    nonfair = _load_ssim(NONFAIR_PATHS)
    for md in (
        RC / "results/resolution_constrained_target_resolution_seed42_b2_10models.md",
        RC / "results/resolution_constrained_fair_seed42_b2_10models.md",
    ):
        if md.is_file():
            patch_md(md, fair)
    patch_nonfair_csv(RC / "logs_nonfair/nonfair_10models_metrics.csv", nonfair)


if __name__ == "__main__":
    main()
