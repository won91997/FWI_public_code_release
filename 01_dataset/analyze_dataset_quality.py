#!/usr/bin/env python3
"""统计 OpenFWI 256 五炮数据集的质量分布。"""
import glob
import os
import sys

import numpy as np

BASE = os.environ.get("DATA_ROOT", "")
ZERO_EPS = 0.0  # 严格按 ==0 判缺失/补零


def iter_samples(base):
    files = sorted(
        glob.glob(os.path.join(base, "seismic_full", "seismic*.npy")),
        key=lambda p: int(os.path.basename(p).replace("seismic", "").replace(".npy", "")),
    )
    gid = 0
    for fp in files:
        arr = np.load(fp, mmap_mode="r")
        for i in range(arr.shape[0]):
            yield gid, fp, i, np.asarray(arr[i], dtype=np.float32)
            gid += 1


def shot_active(shot):
    return bool(np.any(shot != ZERO_EPS))


def trace_coverage(shot):
    # shot: (T, 256) -> 每道是否至少有一个非零样点
    return int(np.count_nonzero(np.any(shot != ZERO_EPS, axis=0)))


def summarize(name, arr):
    arr = np.asarray(arr, dtype=np.float64)
    return {
        "n": len(arr),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def fmt_pct(x):
    return "%.2f%%" % (100.0 * x)


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else BASE
    if not base:
        print("[ERROR] DATA_ROOT is not set and no dataset path was provided.", file=sys.stderr)
        print("Usage: python analyze_dataset_quality.py /path/to/DATA_ROOT", file=sys.stderr)
        sys.exit(2)
    split_path = os.path.join(base, "split_by_inline.npy")
    splits = np.load(split_path, mmap_mode="r") if os.path.isfile(split_path) else None
    split_names = {0: "train", 1: "val", 2: "test"}

    n_samples = 0
    active_shots_per_sample = []
    trace_cov_per_shot = []          # 每个激活炮的道覆盖率 (0~1)
    mean_trace_cov_per_sample = []   # 样本内 5 炮平均道覆盖率
    elem_nonzero_ratio = []          # 样本整体非零体素比例
    slot_cov = []                    # 样本内有效炮-道槽位 / (5*256)
    per_shot_active = np.zeros(5, dtype=np.int64)
    hist_active_shots = np.zeros(6, dtype=np.int64)

    # 分 split 收集
    split_stats = {v: {"active_shots": [], "trace_cov": [], "elem_ratio": []} for v in split_names.values()}

    for gid, fp, row, smp in iter_samples(base):
        nch, nt, nx = smp.shape
        assert nch == 5 and nx == 256

        active = []
        trace_covs = []
        slot_count = 0
        for k in range(5):
            if shot_active(smp[k]):
                active.append(k)
                per_shot_active[k] += 1
                tc_n = trace_coverage(smp[k])
                tc = tc_n / nx
                trace_covs.append(tc)
                trace_cov_per_shot.append(tc)
                slot_count += tc_n

        slot_cov.append(slot_count / (5 * nx))

        n_active = len(active)
        hist_active_shots[n_active] += 1
        active_shots_per_sample.append(n_active)

        if trace_covs:
            mean_trace_cov_per_sample.append(float(np.mean(trace_covs)))
        else:
            mean_trace_cov_per_sample.append(0.0)

        elem_ratio = float(np.count_nonzero(smp)) / smp.size
        elem_nonzero_ratio.append(elem_ratio)

        if splits is not None and gid < len(splits):
            sn = split_names.get(int(splits[gid]))
            if sn:
                split_stats[sn]["active_shots"].append(n_active)
                split_stats[sn]["trace_cov"].append(mean_trace_cov_per_sample[-1])
                split_stats[sn]["elem_ratio"].append(elem_ratio)

        n_samples += 1

    active_shots_per_sample = np.array(active_shots_per_sample, dtype=np.float64)
    mean_trace_cov_per_sample = np.array(mean_trace_cov_per_sample, dtype=np.float64)
    elem_nonzero_ratio = np.array(elem_nonzero_ratio, dtype=np.float64)
    trace_cov_per_shot = np.array(trace_cov_per_shot, dtype=np.float64)
    slot_cov = np.array(slot_cov, dtype=np.float64)

    missing_shot_ratio = 1.0 - active_shots_per_sample / 5.0
    missing_elem_ratio = 1.0 - elem_nonzero_ratio

    print("=" * 72)
    print("Dataset quality summary:", base)
    print("=" * 72)
    print("Total samples:", n_samples)
    print("Shape per sample: (5, 3000, 256)")
    print("Elements per sample:", 5 * 3000 * 256)
    print()

    print("[1] 炮位激活（5 通道 = 5 个连续炮，中心对齐，缺失补零）")
    print("  Per-shot activation count / rate:")
    for k in range(5):
        print(
            "    Shot %d: %5d / %d  (%s)"
            % (k + 1, per_shot_active[k], n_samples, fmt_pct(per_shot_active[k] / n_samples))
        )
    print("  Active shots per sample:")
    s = summarize("active_shots", active_shots_per_sample)
    print(
        "    mean=%.3f  median=%.1f  std=%.3f  min=%d  max=%d"
        % (s["mean"], s["median"], s["std"], int(s["min"]), int(s["max"]))
    )
    print("    p10/p25/p75/p90:", s["p10"], s["p25"], s["p75"], s["p90"])
    print("  Distribution (#active shots -> #samples):")
    for i in range(6):
        print("    %d炮: %5d  (%s)" % (i, hist_active_shots[i], fmt_pct(hist_active_shots[i] / n_samples)))
    print("  Missing shot ratio (1 - active/5): mean=%s  median=%s" % (
        fmt_pct(np.mean(missing_shot_ratio)), fmt_pct(np.median(missing_shot_ratio))
    ))
    print()

    print("[2] 道覆盖率（每炮 256 道，道内任一时间样点非零即计为有数据）")
    print("  Per-sample mean trace coverage (over active shots):")
    s = summarize("trace_cov_sample", mean_trace_cov_per_sample)
    print(
        "    mean=%s  median=%s  std=%s"
        % (fmt_pct(s["mean"]), fmt_pct(s["median"]), fmt_pct(s["std"]))
    )
    print("    p10/p25/p75/p90:", fmt_pct(s["p10"]), fmt_pct(s["p25"]), fmt_pct(s["p75"]), fmt_pct(s["p90"]))
    print("  Per active-shot trace coverage:")
    s2 = summarize("trace_cov_shot", trace_cov_per_shot)
    print(
        "    mean=%s  median=%s  std=%s  (n=%d active shots)"
        % (fmt_pct(s2["mean"]), fmt_pct(s2["median"]), fmt_pct(s2["std"]), s2["n"])
    )
    print("    p10/p25/p75/p90:", fmt_pct(s2["p10"]), fmt_pct(s2["p25"]), fmt_pct(s2["p75"]), fmt_pct(s2["p90"]))
    print()

    print("[3] 体素非零率（5×3000×256 全体素，==0 视为缺失/补零）")
    s = summarize("elem_ratio", elem_nonzero_ratio)
    print(
        "    mean=%s  median=%s  std=%s"
        % (fmt_pct(s["mean"]), fmt_pct(s["median"]), fmt_pct(s["std"]))
    )
    print("    p10/p25/p75/p90:", fmt_pct(s["p10"]), fmt_pct(s["p25"]), fmt_pct(s["p75"]), fmt_pct(s["p90"]))
    print("  Missing (zero) ratio:")
    print(
        "    mean=%s  median=%s"
        % (fmt_pct(np.mean(missing_elem_ratio)), fmt_pct(np.median(missing_elem_ratio)))
    )
    print()

    print("[4] 综合有效覆盖率（炮位×道，按 5×256 个炮-道槽位）")
    s = summarize("slot_cov", slot_cov)
    print(
        "    mean=%s  median=%s  std=%s"
        % (fmt_pct(s["mean"]), fmt_pct(s["median"]), fmt_pct(s["std"]))
    )
    print("    p10/p25/p75/p90:", fmt_pct(s["p10"]), fmt_pct(s["p25"]), fmt_pct(s["p75"]), fmt_pct(s["p90"]))
    print()

    if splits is not None:
        print("[5] 按 split 划分 (train/val/test by inline)")
        for sn in ("train", "val", "test"):
            st = split_stats[sn]
            if not st["active_shots"]:
                continue
            a = np.array(st["active_shots"], dtype=np.float64)
            t = np.array(st["trace_cov"], dtype=np.float64)
            e = np.array(st["elem_ratio"], dtype=np.float64)
            print(
                "  %s: n=%4d | active_shots mean/median=%.2f/%.1f | trace_cov mean/median=%s/%s | elem_nonzero mean/median=%s/%s"
                % (
                    sn,
                    len(a),
                    np.mean(a),
                    np.median(a),
                    fmt_pct(np.mean(t)),
                    fmt_pct(np.median(t)),
                    fmt_pct(np.mean(e)),
                    fmt_pct(np.median(e)),
                )
            )
        print()

    print("[6] 制作参数 (from make_dataset.log / dataset_meta.txt)")
    meta = os.path.join(base, "dataset_meta.txt")
    if os.path.isfile(meta):
        with open(meta) as f:
            for line in f:
                print(" ", line.rstrip())
    print("  min_xl_coverage=0.70 (允许单炮缺最多30%%道，补零)")
    print("  allow_partial=True (邻炮缺失整炮补零)")
    print("=" * 72)


if __name__ == "__main__":
    main()
