#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
炮集 (Shot Gather) 数据集 — OpenFWI 标准复现版。
核心：5 通道 = 5 个连续炮，中间炮对齐，邻炮缺失补零 (--allow-partial)。
用法（复现论文）:
  python3 make_shot_gather_dataset.py --seismic-dir <SEISMIC_DIR> --velocity VELDAT_xxx.SEGY --out out \\
    --five-channels --depth-range 10000 --allow-partial
默认: n_crossline=256, time_points=3000。可选: pip install tqdm 显示进度。
"""
import argparse
import os
import struct
import sys
from collections import defaultdict

try:
    import segyio
    import numpy as np
except ImportError as e:
    print("Need: pip install segyio numpy")
    sys.exit(1)
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None, **kw):
        return iterable

# 道头（1-based）
FLDR, TRACF = 9, 13
EP_IL, CDP_XL = 17, 21
OFFSET = 37
SX, SY = 73, 77
GX, GY = 81, 85  # Receiver group X/Y，用于跨炮匹配（EP_IL/CDP_XL 可能为 midpoint）
VEL_IL, VEL_XL = 189, 193

INLINE_MIN, INLINE_MAX = 600, 894
CROSSLINE_MIN, CROSSLINE_MAX = 1200, 2200
XL_FULL_MIN, XL_FULL_MAX = 1100, 3900
N_CROSSLINE_DEFAULT = 256  # 复现野外数据默认 256 道
# Offset-Sectored 模式：5 个 Offset 分组 (m)，[0-1300), [1300-2600), [2600-3900), [3900-5200), [5200-6500)
OFFSET_BINS = [0, 1300, 2600, 3900, 5200, 6500]
FULL_TIME_POINTS = 3000  # 6s 全波形
FULL_DEPTH_POINTS = 256
SAMPLES_PER_FILE = 500


def read_int4(buf, byte_1based):
    off = byte_1based - 1
    return struct.unpack(">i", buf[off : off + 4])[0]


def _shot_key(buf, shot_by):
    fldr = read_int4(buf, FLDR)
    ep = read_int4(buf, EP_IL)
    cdp = read_int4(buf, CDP_XL)
    sx = read_int4(buf, SX)
    sy = read_int4(buf, SY)
    if shot_by == "fldr":
        return fldr
    if shot_by == "sx_sy":
        return (sx, sy)
    if shot_by == "ep_cdp":
        return (ep, cdp)
    raise ValueError("shot_by must be fldr | sx_sy | ep_cdp")


def build_shot_to_traces(seismic_path, shot_by):
    """
    单文件：从地震 SEG-Y 按 shot_by 分组，返回 shot_key -> [(offset, file_idx, trace_idx, il, xl), ...]（file_idx=0），已按 offset 排序。
    """
    with segyio.open(seismic_path, "r", ignore_geometry=True) as f:
        group = defaultdict(list)
        for i in range(f.tracecount):
            buf = f.header[i].buf
            key = _shot_key(buf, shot_by)
            off = read_int4(buf, OFFSET)
            il = read_int4(buf, EP_IL)
            xl = read_int4(buf, CDP_XL)
            group[key].append((off, 0, i, il, xl))
        for key in group:
            group[key].sort(key=lambda x: x[0])
        return dict(group), len(f.samples)


def build_shot_to_traces_multi(seismic_paths, shot_by, take_first_70=True, target_offsets=None, n_crossline=N_CROSSLINE_DEFAULT):
    """
    多文件：按 shot 合并所有文件中的道。
    take_first_70=True：按 offset 排序后每组只保留前 n_crossline 道（默认）。
    target_offsets=None：不启用固定偏移距；若为长度 n_crossline 的数组，则每炮按目标 offset 各选最近的一道。
    返回 shot_key -> [(offset, file_idx, trace_idx, il, xl), ...], n_samples。
    """
    merged = defaultdict(list)
    n_samples = None
    for file_idx, path in enumerate(seismic_paths):
        with segyio.open(path, "r", ignore_geometry=True) as f:
            if n_samples is None:
                n_samples = len(f.samples)
            for i in range(f.tracecount):
                buf = f.header[i].buf
                key = _shot_key(buf, shot_by)
                off = read_int4(buf, OFFSET)
                il = read_int4(buf, EP_IL)
                xl = read_int4(buf, CDP_XL)
                merged[key].append((off, file_idx, i, il, xl))
    for key in merged:
        merged[key].sort(key=lambda x: x[0])
        if target_offsets is not None and len(target_offsets) == n_crossline:
            merged[key] = _select_by_target_offsets(merged[key], target_offsets)
        elif take_first_70:
            merged[key] = merged[key][:n_crossline]
    return dict(merged), n_samples


def build_shot_to_traces_full(seismic_paths, shot_by, receiver_by="ep_cdp"):
    """按炮合并所有道（不截断）。receiver_by: ep_cdp=用(EP_IL,CDP_XL)匹配；gx_gy=用(GX,GY)检波器坐标匹配。
    返回 shot_key -> [(offset, file_idx, trace_idx, il, xl, rx, ry), ...]，(rx,ry)为 receiver 查找键。"""
    merged = defaultdict(list)
    n_samples = None
    print("Scanning %d SEG-Y files (indexing headers)..." % len(seismic_paths))
    for file_idx, path in enumerate(tqdm(seismic_paths, desc="Indexing")):
        with segyio.open(path, "r", ignore_geometry=True) as f:
            if n_samples is None:
                n_samples = len(f.samples)
            for i in range(f.tracecount):
                buf = f.header[i].buf
                key = _shot_key(buf, shot_by)
                off = read_int4(buf, OFFSET)
                il = read_int4(buf, EP_IL)
                xl = read_int4(buf, CDP_XL)
                gx = read_int4(buf, GX)
                gy = read_int4(buf, GY)
                rx, ry = (gx, gy) if receiver_by == "gx_gy" else (il, xl)
                merged[key].append((off, file_idx, i, il, xl, rx, ry))
    print("Sorting traces per shot...")
    for key in merged:
        merged[key].sort(key=lambda x: x[0])
    return dict(merged), n_samples


def _shot_to_ilxl_lookup(shot_to_traces, use_rx_ry=False):
    """为每炮构建 receiver_key -> (file_idx, trace_idx, offset) 查找表。
    use_rx_ry: True 时用 (rx,ry)，否则用 (il,xl)。trace 格式为 (off,fidx,ti,il,xl,rx,ry)。"""
    out = {}
    for shot_key, lst in shot_to_traces.items():
        m = {}
        for t in lst:
            key = (t[5], t[6]) if use_rx_ry and len(t) >= 7 else (t[3], t[4])
            m[key] = (t[1], t[2], t[0])
        out[shot_key] = m
    return out


def build_ilxl_to_traces(seismic_paths):
    """
    Offset-Sectored 模式：按 (il, xl) 索引所有道。
    返回 (il, xl) -> [(offset, file_idx, trace_idx), ...], n_samples。
    用于遍历所有 87 个 offset 文件，在同一 (il,xl) 下按 offset 分到 5 个通道。
    """
    ilxl = defaultdict(list)
    n_samples = None
    for file_idx, path in enumerate(seismic_paths):
        with segyio.open(path, "r", ignore_geometry=True) as f:
            if n_samples is None:
                n_samples = len(f.samples)
            for i in range(f.tracecount):
                buf = f.header[i].buf
                il = read_int4(buf, EP_IL)
                xl = read_int4(buf, CDP_XL)
                off = read_int4(buf, OFFSET)
                ilxl[(il, xl)].append((off, file_idx, i))
    return dict(ilxl), n_samples


def _offset_bin_index(offset, bins):
    """offset (m) 落在哪个 bin，返回 0..4；超出范围返回 -1。"""
    for k in range(len(bins) - 1):
        if bins[k] <= offset < bins[k + 1]:
            return k
    return -1


def _debug_why_zero_windows(center_fldr, trace_list, xl_starts, n_crossline, il_min, il_max, xl_min, xl_max, vel_trace_idx_fn, ntr_vel):
    """诊断为何某炮的 _extract_windows_from_shot 返回空"""
    xl2best = {}
    for t in trace_list:
        off, fidx, tidx, il, xl = t[0], t[1], t[2], t[3], t[4]
        if xl not in xl2best or off < xl2best[xl][0]:
            xl2best[xl] = t
    xls = sorted(xl2best.keys())
    print("\n  [DEBUG] Shot %s: center has %d traces, xl coverage: %d..%d (%d unique xl)" % (
        center_fldr, len(trace_list), min(xls) if xls else 0, max(xls) if xls else 0, len(xls)))
    if len(xls) < n_crossline:
        print("  [DEBUG] FAIL: need %d consecutive xl, only have %d unique xl" % (n_crossline, len(xls)))
        return
    # 找最长连续 xl 段
    max_run = 0
    max_start = xls[0] if xls else 0
    run_start = 0
    for j in range(1, len(xls) + 1):
        if j < len(xls) and xls[j] == xls[j-1] + 1:
            continue
        run_len = j - run_start
        if run_len > max_run:
            max_run = run_len
            max_start = xls[run_start]
        run_start = j
    print("  [DEBUG] Longest consecutive xl run: %d (xl %d..%d)" % (max_run, max_start, max_start + max_run - 1 if max_run else 0))
    if max_run < n_crossline:
        print("  [DEBUG] FAIL: need %d consecutive, max run=%d" % (n_crossline, max_run))
    # 采样第一个 xl_start 检查
    xs = xl_starts[0] if xl_starts else xl_min
    for xl in range(xs, min(xs + n_crossline, xs + 100)):
        if xl not in xl2best:
            print("  [DEBUG] First missing xl at xl_start=%d: xl=%d not in xl2best" % (xs, xl))
            break
        t = xl2best[xl]
        il = t[3]
        idx = vel_trace_idx_fn(il, xl)
        if il < il_min or il > il_max:
            print("  [DEBUG] At xl=%d: il=%d out of range [%d,%d]" % (xl, il, il_min, il_max))
            break
        if idx < 0 or idx >= ntr_vel:
            print("  [DEBUG] At xl=%d,il=%d: vel_trace_idx=%d out of [0,%d)" % (xl, il, idx, ntr_vel))
            break


def _debug_why_fail_all_have(group, positions, shot_to_ilxl, receiver_by):
    """诊断为何某炮的 windows 全部 fail all_have。positions 为 [(il,xl,rx,ry),...]，缺道时 rx,ry 为 None"""
    use_rx = receiver_by == "gx_gy"
    print("\n  [DEBUG] Group %s: all_have failed. Sample positions (first 3): %s" % (group[:3], positions[:3]))
    for sk in group:
        m = shot_to_ilxl.get(sk, {})
        missing = []
        for pos in positions[:5]:
            key = (pos[2], pos[3]) if use_rx and len(pos) >= 4 else (pos[0], pos[1])
            if (use_rx and len(pos) >= 4 and (pos[2] is None or pos[3] is None)):
                continue
            if key not in m:
                missing.append(key)
        if missing:
            print("  [DEBUG] Shot %s: missing %d positions in lookup, e.g. %s" % (sk, len([p for p in positions if ((p[2],p[3]) if use_rx else (p[0],p[1])) not in m]), missing[:2]))


def _extract_windows_from_shot(trace_list, xl_starts, n_crossline, il_min, il_max, xl_min, xl_max, vel_trace_idx_fn, ntr_vel, min_xl_coverage=1.0):
    """
    从一炮的 trace_list 中，按 xl_starts 提取多个窗口。每窗口需 n_crossline 道，xl 连续 [xl_start, xl_start+n_crossline-1]。
    min_xl_coverage: 最低覆盖率，如 0.7 表示允许缺 30% 道（补零）。1.0=必须全部有道。
    返回 [(trace_subset, il_ref, xl_start), ...]，trace_subset 为长 n_crossline 的列表，每项为 trace 或 None（缺道补零）。
    """
    if len(trace_list) < 1:
        return []
    min_found = max(1, int(n_crossline * min_xl_coverage))
    xl2best = {}
    for t in trace_list:
        off, fidx, tidx, il, xl = t[0], t[1], t[2], t[3], t[4]
        if xl not in xl2best or off < xl2best[xl][0]:
            xl2best[xl] = t
    out = []
    for xl_start in xl_starts:
        xl_end = xl_start + n_crossline
        if xl_start < xl_min or xl_end - 1 > xl_max:
            continue
        window = []  # 长度 n_crossline，每项 trace 或 None
        il_ref = None
        for xl in range(xl_start, xl_end):
            if xl in xl2best:
                t = xl2best[xl]
                il = t[3]
                if il_min <= il <= il_max:
                    idx = vel_trace_idx_fn(il, xl)
                    if 0 <= idx < ntr_vel:
                        window.append(t)
                        if il_ref is None:
                            il_ref = il
                        continue
            window.append(None)
        if len([x for x in window if x is not None]) >= min_found and il_ref is not None:
            out.append((window, il_ref, xl_start))
    return out


def _select_by_target_offsets(trace_list, target_offsets):
    """
    从 trace_list（已按 offset 排序）中为每个 target_offsets[i] 选一道：选与目标距离最近且尚未被选中的道。
    若某目标下无未选道则跳过该目标；调用方应只保留 len==len(target_offsets) 的炮。
    """
    n_want = len(target_offsets)
    if len(trace_list) < n_want:
        return []
    used = [False] * len(trace_list)
    out = []
    for target in target_offsets:
        best_j, best_d = None, None
        for j in range(len(trace_list)):
            if used[j]:
                continue
            d = abs(trace_list[j][0] - target)
            if best_d is None or d < best_d:
                best_d = d
                best_j = j
        if best_j is not None:
            used[best_j] = True
            out.append(trace_list[best_j])
    return out


def load_offset_list(path, expected_len=None):
    """从文件读取一行一个的整数（跳过 # 和空行），返回长度 expected_len 的列表（若 expected_len 非 None 则校验）。"""
    vals = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for part in line.replace(",", " ").split():
                try:
                    vals.append(int(part))
                except ValueError:
                    continue
    if expected_len is not None and len(vals) != expected_len:
        raise ValueError("offset list file %s: got %d values, expected %d" % (path, len(vals), expected_len))
    return vals


def load_velocity_trace_data(velocity_path, depth_points, depth_range_m=None, depth_interval_m=10):
    """
    加载速度模型。depth_range_m: 若指定，则从原始全深度重采样到 depth_points 覆盖 0~depth_range_m。
    depth_interval_m: 原始速度的深度采样间隔(米)，默认 10。若非 10m 请用 --depth-interval 指定。
    """
    with segyio.open(velocity_path, "r", ignore_geometry=True) as f:
        ntr = f.tracecount
        n_raw = len(f.samples)
        if depth_range_m is not None and n_raw * depth_interval_m > depth_range_m:
            data = np.zeros((ntr, depth_points), dtype=np.float32)
            n_use = min(n_raw, int(depth_range_m / depth_interval_m) + 1)
            depth_old = np.arange(n_use, dtype=np.float32) * float(depth_interval_m)
            depth_new = np.linspace(0, min((n_use - 1) * depth_interval_m, depth_range_m), depth_points, dtype=np.float32)
            for i in range(ntr):
                tr = np.asarray(f.trace[i][:n_use], dtype=np.float32)
                data[i] = np.interp(depth_new, depth_old, tr) if n_use > 1 else np.full(depth_points, tr[0])
            return data
        data = np.zeros((ntr, depth_points), dtype=np.float32)
        for i in range(ntr):
            data[i] = f.trace[i][:depth_points]
        return data


def run(seismic_path=None, seismic_paths=None, velocity_path=None, out_dir="running/TrainDatasets_shot", shot_by="fldr", il_range=(INLINE_MIN, INLINE_MAX), xl_range=(CROSSLINE_MIN, CROSSLINE_MAX), max_samples=None, count_only=False, fixed_offset_targets=False, offset_min=0, offset_max=5175, offset_list_path=None, time_points=3000, downsample=False, depth_points=256, depth_range_m=None, depth_interval_m=10, velocity_per_trace=True, five_channels=False, sliding_window=False, xl_starts=None, shot_decimation=1, offset_sectored=False, n_crossline=None, allow_partial=False, min_xl_coverage=0.7, save_positions=False, skip_shot_gap=None, debug_five_shot=False, debug_discard=False, receiver_by="ep_cdp", flip_flop=False):
    n_cross = n_crossline if n_crossline is not None else N_CROSSLINE_DEFAULT
    if offset_sectored and (five_channels or sliding_window):
        raise ValueError("--offset-sectored 与 --five-channels / --sliding-window 互斥，请只选其一")
    if seismic_paths is not None:
        paths = list(seismic_paths)
    elif seismic_path is not None:
        paths = [seismic_path]
    else:
        raise ValueError("need --seismic or --seismic-files")
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError("地震文件不存在（请确认当前工作目录或使用绝对路径）: %s\n当前目录: %s" % (missing, os.path.abspath(os.curdir)))
    if not os.path.isfile(velocity_path):
        raise FileNotFoundError("速度文件不存在: %s\n当前目录: %s" % (velocity_path, os.path.abspath(os.curdir)))
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "seismic_full"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "vmodel_full"), exist_ok=True)

    print("Shot key: %s" % shot_by)
    print("n_crossline: %d" % n_cross)
    target_offsets = None
    if fixed_offset_targets:
        if offset_list_path and os.path.isfile(offset_list_path):
            target_offsets = load_offset_list(offset_list_path, n_cross)
            print("Fixed offset targets: 70 values from file %s (min=%d max=%d)" % (offset_list_path, min(target_offsets), max(target_offsets)))
        else:
            target_offsets = np.linspace(offset_min, offset_max, n_cross, dtype=np.int32).tolist()
            print("Fixed offset targets: 70 values from %d to %d (same offset meaning per column across shots)" % (offset_min, offset_max))
    five_channel_mode = five_channels
    offset_sectored_mode = offset_sectored
    shot_to_ilxl = None  # 5-shot 模式用：(il,xl) 查找
    if offset_sectored_mode:
        ilxl_to_traces, _ = build_ilxl_to_traces(paths)
        print("Seismic: Offset-Sectored, %d files, index by (il,xl), 5 channels = 5 offset bins %s" % (len(paths), OFFSET_BINS))
    elif five_channel_mode:
        shot_to_traces, _ = build_shot_to_traces_full(paths, shot_by, receiver_by=receiver_by)
        shot_to_ilxl = _shot_to_ilxl_lookup(shot_to_traces, use_rx_ry=(receiver_by == "gx_gy"))
        print("Seismic: 5-shot mode (OpenFWI), %d files, center-shot aligned, (5, T, %d), receiver_by=%s%s" % (len(paths), n_cross, receiver_by, ", flip_flop" if flip_flop else ""))
    elif sliding_window:
        shot_to_traces, _ = build_shot_to_traces_full(paths, shot_by, receiver_by="ep_cdp")
        xl_starts_list = xl_starts if xl_starts else [1200, 1400, 1600, 1800, 2000]
        print("Seismic: 2.5D sliding window, %d files, all traces per shot" % len(paths))
        print("  xl_starts=%s, %d windows/shot, shot_decimation=%d" % (xl_starts_list, len(xl_starts_list), shot_decimation))
    elif len(paths) == 1:
        shot_to_traces, _ = build_shot_to_traces(paths[0], shot_by)
        print("Seismic: single file %s" % paths[0])
        if fixed_offset_targets:
            print("(fixed_offset_targets ignored for single file; use --seismic-files for fixed offsets)")
    else:
        shot_to_traces, _ = build_shot_to_traces_multi(paths, shot_by, take_first_70=not fixed_offset_targets, target_offsets=target_offsets, n_crossline=n_cross)
        if fixed_offset_targets:
            print("Seismic: %d files merged by shot, %d traces per shot by nearest-to-target offset" % (len(paths), n_cross))
        else:
            print("Seismic: %d files merged by shot, take first %d traces by offset" % (len(paths), n_cross))
    il_min, il_max = il_range
    xl_min, xl_max = xl_range

    # 速度取法：velocity_per_trace=True 时每道用自身的 (il,xl) 抽速度（逐道对应，可不连续）；否则用第一道 (il,xl) 取连续 70 道
    if velocity_per_trace:
        print("Velocity: per-trace (il,xl) -> each column c = vel(il_c, xl_c), 70 (il,xl) may be non-consecutive")
    vel_data = load_velocity_trace_data(velocity_path, max(depth_points, FULL_DEPTH_POINTS), depth_range_m=depth_range_m, depth_interval_m=depth_interval_m)
    ntr_vel = vel_data.shape[0]

    def vel_trace_idx(il, xl):
        return (il - 600) * 2801 + (xl - 1100)

    shots_valid = []
    offset_sectored_windows = []
    five_shot_samples = []  # [(shot_keys[5], [(il,xl), ...]), ...]，中间炮对齐
    if offset_sectored_mode:
        # 枚举有效窗口：(il, xl_start) 满足 xl_start..xl_start+n_cross-1 在 ilxl 中均有道
        if xl_starts:
            xl_starts_list = xl_starts
        else:
            step = max(1, n_cross // 2)  # 50% 重叠滑窗
            xl_starts_list = list(range(xl_min, xl_max - n_cross + 2, step))
        if not xl_starts_list:
            xl_starts_list = [xl_min]
        for il in range(il_min, il_max + 1):
            for xl_start in xl_starts_list:
                xl_end = xl_start + n_cross
                if xl_start < xl_min or xl_end - 1 > xl_max:
                    continue
                window = []
                for xl in range(xl_start, xl_end):
                    key = (il, xl)
                    if key not in ilxl_to_traces:
                        break
                    tracelist = ilxl_to_traces[key]
                    idx = vel_trace_idx(il, xl)
                    if idx < 0 or idx >= ntr_vel:
                        break
                    window.append((xl, tracelist))
                if len(window) == n_cross:
                    offset_sectored_windows.append((il, xl_start, window))
        if max_samples is not None:
            offset_sectored_windows = offset_sectored_windows[:max_samples]
        print("Offset-Sectored valid windows: %d (il, xl_start) with %d traces each" % (len(offset_sectored_windows), n_cross))
    elif five_channel_mode:
        # 5 炮模式：以中间炮为基准，5 通道必须同一组 (il,xl)，速度取中间炮正下方
        # flip_flop=True：奇偶震源，取 i-4,i-2,i,i+2,i+4（步长2），否则 i-2,i-1,i,i+1,i+2
        shot_keys = sorted(shot_to_traces.keys())
        if shot_decimation > 1:
            shot_keys = shot_keys[::shot_decimation]
        xl_starts_list = xl_starts if xl_starts else list(range(xl_min, xl_max - n_cross + 2, max(1, n_cross // 2)))
        if not xl_starts_list:
            xl_starts_list = [xl_min]
        n_skipped_gap = 0
        step = 2 if flip_flop else 1
        half = 2 * step  # 左/右各 2 个邻居
        # flip_flop 时同线取炮，FLDR 跳号+decimation 导致 gap 很大，不检查 gap
        eff_skip_gap = None if flip_flop else skip_shot_gap
        if flip_flop:
            print("  (flip_flop: skip_shot_gap disabled, same-line shots)")
        if min_xl_coverage < 1.0:
            pct_missing = int((1 - min_xl_coverage) * 100)
            print("  (min_xl_coverage=%.2f: allow up to %d%% missing traces, zero-pad)" % (min_xl_coverage, pct_missing))
        n_zero_windows = 0
        n_fail_all_have = 0
        debug_sample_i = len(shot_keys) // 2 if debug_discard else -1
        for i in tqdm(range(half, len(shot_keys) - half), desc="Planning 5-shot"):
            group = tuple(shot_keys[i - half : i + half + 1 : step])
            if eff_skip_gap is not None and shot_by == "fldr":
                gap_ok = True
                for k in range(4):
                    if abs(group[k + 1] - group[k]) > eff_skip_gap:
                        gap_ok = False
                        n_skipped_gap += 1
                        break
                if not gap_ok:
                    continue
            center_lst = shot_to_traces[group[2]]
            windows = _extract_windows_from_shot(center_lst, xl_starts_list, n_cross, il_min, il_max, xl_min, xl_max, vel_trace_idx, ntr_vel, min_xl_coverage=min_xl_coverage)
            if len(windows) == 0:
                n_zero_windows += 1
                if debug_discard and i == debug_sample_i:
                    _debug_why_zero_windows(group[2], center_lst, xl_starts_list, n_cross, il_min, il_max, xl_min, xl_max, vel_trace_idx, ntr_vel)
            n_added_this_group = 0
            for trace_subset, il_ref, xl_start in windows:
                positions = []
                for c, t in enumerate(trace_subset):
                    if t is not None:
                        positions.append((t[3], t[4], t[5], t[6]))  # (il, xl, rx, ry)
                    else:
                        positions.append((il_ref, xl_start + c, None, None))  # 缺道，velocity 用 (il_ref, xl_start+c)
                all_have = True
                for sk in group:
                    m = shot_to_ilxl.get(sk, {})
                    for pos in positions:
                        rx, ry = (pos[2], pos[3]) if len(pos) >= 4 else (pos[0], pos[1])
                        if rx is None or ry is None:
                            continue
                        if (rx, ry) not in m:
                            all_have = False
                            break
                    if not all_have:
                        break
                if all_have or allow_partial:
                    five_shot_samples.append((group, positions))
                    n_added_this_group += 1
            if len(windows) > 0 and n_added_this_group == 0:
                n_fail_all_have += 1
                if debug_discard and i == debug_sample_i:
                    ts, ir, xs = windows[0]
                    pos0 = [(t[3], t[4], t[5], t[6]) if t else (ir, xs+c, None, None) for c, t in enumerate(ts)]
                    _debug_why_fail_all_have(group, pos0, shot_to_ilxl, receiver_by)
        if debug_discard:
            print("  [DEBUG-DISCARD] Groups with 0 windows (from _extract_windows): %d" % n_zero_windows)
            print("  [DEBUG-DISCARD] Groups with windows but all fail all_have check: %d" % n_fail_all_have)
        if max_samples is not None:
            five_shot_samples = five_shot_samples[:max_samples]
        if n_skipped_gap > 0:
            print("  (skipped %d groups: shot gap > %d)" % (n_skipped_gap, eff_skip_gap))
        print("5-shot valid samples (center aligned%s): %d" % (", allow_partial" if allow_partial else "", len(five_shot_samples)))
    elif sliding_window:
        xl_starts_list = xl_starts if xl_starts else [1200, 1400, 1600, 1800, 2000]
        shot_keys = sorted(shot_to_traces.keys())
        if shot_decimation > 1:
            shot_keys = shot_keys[::shot_decimation]
        for idx, shot_key in enumerate(shot_keys):
            lst = shot_to_traces[shot_key]
            windows = _extract_windows_from_shot(lst, xl_starts_list, n_cross, il_min, il_max, xl_min, xl_max, vel_trace_idx, ntr_vel, min_xl_coverage=1.0)
            for trace_subset, il_ref, _ in windows:
                trace_list = [t for t in trace_subset if t is not None]
                if len(trace_list) < n_cross:
                    continue
                shots_valid.append((shot_key, trace_list, il_ref, None))
    else:
        for shot_key, lst in shot_to_traces.items():
            lst = lst[:n_cross]
            if len(lst) < n_cross:
                continue
            if velocity_per_trace:
                all_in = True
                for _, _, _, il, xl in lst:
                    if il_min <= il <= il_max and xl_min <= xl <= xl_max:
                        idx = vel_trace_idx(il, xl)
                        if idx < 0 or idx >= ntr_vel:
                            all_in = False
                            break
                    else:
                        all_in = False
                        break
                if all_in:
                    shots_valid.append((shot_key, lst, None, None))
            else:
                il, xl = lst[0][3], lst[0][4]
                if il_min <= il <= il_max and xl_min <= xl <= xl_max and xl + n_cross - 1 <= xl_max:
                    idx0 = vel_trace_idx(il, xl)
                    idx1 = vel_trace_idx(il, xl + n_cross - 1)
                    if 0 <= idx0 and idx1 < ntr_vel:
                        shots_valid.append((shot_key, lst, il, xl))

    if not five_channel_mode:
        if max_samples is not None:
            shots_valid = shots_valid[:max_samples]
        if sliding_window:
            print("Valid samples (2.5D sliding): %d" % len(shots_valid))
        else:
            print("Valid shots (>= %d traces, (il,xl) in vel range): %d" % (n_cross, len(shots_valid)))

    if count_only:
        print("(--count-only: skip writing, exit.)")
        if offset_sectored_mode:
            return len(offset_sectored_windows)
        if five_channel_mode:
            return len(five_shot_samples)
        return len(shots_valid)

    # 迭代列表
    if offset_sectored_mode:
        iter_list = offset_sectored_windows  # 每项为 (il, xl_start, window)
    elif five_channel_mode:
        iter_list = five_shot_samples  # 每项为 (group[5], positions[(il,xl),...])
    else:
        iter_list = [(s,) for s in shots_valid]  # 每项为 (shot, trace_list, il_ref, xl_first)

    # 起飞前核验：磁盘空间预估 (float32=4 bytes)
    n_total = len(iter_list)
    n_ch = 5 if (five_channel_mode or offset_sectored_mode) else 1
    bytes_seis = n_total * n_ch * time_points * n_cross * 4
    bytes_vel = n_total * 1 * depth_points * n_cross * 4
    est_gb = (bytes_seis + bytes_vel) / (1024**3)
    print("[Pre-flight] Est. disk: %.1f GB for %d samples (seismic %.1f + velocity %.1f)" % (est_gb, n_total, bytes_seis / (1024**3), bytes_vel / (1024**3)))
    if est_gb > 100 and max_samples is None:
        print("[Pre-flight] WARNING: >100GB. Consider --max-samples 5000 for a test run first.")

    samples_seis = []
    samples_vel = []
    sample_il_list = []
    samples_positions = []  # (n_cross, 2) per sample, for alignment verification
    out_file_idx = 1
    total = 0

    # 输出时间点数：time_points；深度点数 depth_points
    if depth_range_m is not None:
        print("Velocity: %d points, resampled 0~%d m (interval=%.1fm)" % (depth_points, depth_range_m, depth_interval_m))
    elif depth_points != FULL_DEPTH_POINTS:
        print("Velocity depth points: %d (truncate to first %d, ~10 m/point -> 0~%d m)" % (depth_points, depth_points, (depth_points - 1) * 10))
    max_read = max(time_points, 3000) if downsample else time_points
    if time_points != FULL_TIME_POINTS or downsample:
        print("Seismic time points: %d (max read %d%s)" % (time_points, max_read, ", downsampled" if downsample else ""))
    seismic_handles = [segyio.open(p, "r", ignore_geometry=True) for p in paths]
    try:
        for item in tqdm(iter_list, desc="Writing"):
            if offset_sectored_mode:
                # item = (il, xl_start, window), window = [(xl, [(off, file_idx, ti), ...]), ...]
                il_win, xl_start, window = item
                traces_per_ch = defaultdict(lambda: defaultdict(list))  # (ch, c) -> [trace_array]
                for c, (xl, tracelist) in enumerate(window):
                    for (off, file_idx, ti) in tracelist:
                        ch = _offset_bin_index(off, OFFSET_BINS)
                        if ch < 0:
                            continue
                        tr = np.asarray(seismic_handles[file_idx].trace[ti], dtype=np.float32)
                        traces_per_ch[ch][c].append(tr)
                patch = np.zeros((5, time_points, n_cross), dtype=np.float32)
                for ch in range(5):
                    for c in range(n_cross):
                        lst = traces_per_ch[ch].get(c, [])
                        if not lst:
                            continue
                        stacked = np.mean(lst, axis=0)
                        n_read = min(len(stacked), max_read)
                        if downsample and n_read > time_points:
                            idx = np.linspace(0, n_read - 1, time_points, dtype=np.int64)
                            patch[ch, :, c] = stacked[idx]
                        else:
                            ln = min(n_read, time_points)
                            patch[ch, :ln, c] = stacked[:ln]
                patch_vel = np.zeros((1, depth_points, n_cross), dtype=np.float32)
                for c, (xl, _) in enumerate(window):
                    idx = vel_trace_idx(il_win, xl)
                    patch_vel[0, :, c] = vel_data[idx][:depth_points]
                samples_seis.append(patch[np.newaxis, ...])
                samples_vel.append(patch_vel)
                sample_il_list.append(il_win)
                if save_positions:
                    samples_positions.append(np.array([[il_win, xl] for xl, _ in window], dtype=np.int32))
            elif five_channel_mode:
                # item = (group[5], positions[(il,xl,rx,ry),...])，中间炮 group[2] 对齐，速度取(il,xl)，地震用(rx,ry)跨炮匹配
                # allow_partial 时缺失道填 0
                group, positions = item
                patch = np.zeros((5, time_points, n_cross), dtype=np.float32)
                hits = [0] * 5
                for ch in range(5):
                    ilxl_map = shot_to_ilxl.get(group[ch], {})
                    for c, pos in enumerate(positions):
                        il_c, xl_c = pos[0], pos[1]
                        rx, ry = (pos[2], pos[3]) if len(pos) >= 4 else (il_c, xl_c)
                        tup = ilxl_map.get((rx, ry))
                        if tup is None:
                            continue  # 保持 0
                        hits[ch] += 1
                        fidx, ti, _ = tup
                        tr = np.asarray(seismic_handles[fidx].trace[ti], dtype=np.float32)
                        n_read = min(len(tr), max_read)
                        if downsample and n_read > time_points:
                            idx = np.linspace(0, n_read - 1, time_points, dtype=np.int64)
                            patch[ch, :, c] = tr[idx]
                        else:
                            ln = min(n_read, time_points)
                            patch[ch, :ln, c] = tr[:ln]
                if debug_five_shot and total < 5:
                    print("[DEBUG 5-shot] sample %d: group=%s hits Ch0=%d Ch1=%d Ch2=%d Ch3=%d Ch4=%d (of %d)" % (total, group, hits[0], hits[1], hits[2], hits[3], hits[4], len(positions)))
                samples_seis.append(patch[np.newaxis, ...])
                patch_vel = np.zeros((1, depth_points, n_cross), dtype=np.float32)
                for c, pos in enumerate(positions):
                    il_c, xl_c = pos[0], pos[1]
                    idx = vel_trace_idx(il_c, xl_c)
                    patch_vel[0, :, c] = vel_data[idx][:depth_points]
                samples_vel.append(patch_vel)
                sample_il_list.append(positions[0][0])
                if save_positions:
                    samples_positions.append(np.array([[p[0], p[1]] for p in positions], dtype=np.int32))
            else:
                (shot_key, trace_list, il_first, xl_first) = item[0]
                if xl_first is None and trace_list:
                    xl_first = trace_list[0][4]
                patch = np.zeros((time_points, n_cross), dtype=np.float32)
                for c, (off, file_idx, ti, il_c, xl_c) in enumerate(trace_list):
                    tr = np.asarray(seismic_handles[file_idx].trace[ti], dtype=np.float32)
                    n_read = min(len(tr), max_read)
                    if downsample and n_read > time_points:
                        idx = np.linspace(0, n_read - 1, time_points, dtype=np.int64)
                        patch[:, c] = tr[idx]
                    else:
                        ln = min(n_read, time_points)
                        patch[:ln, c] = tr[:ln]
                samples_seis.append(patch[np.newaxis, ...])
                patch_vel = np.zeros((1, depth_points, n_cross), dtype=np.float32)
                if velocity_per_trace:
                    for c, (_, _, _, il_c, xl_c) in enumerate(trace_list):
                        idx = vel_trace_idx(il_c, xl_c)
                        patch_vel[0, :, c] = vel_data[idx][:depth_points]
                else:
                    for c in range(n_cross):
                        idx = vel_trace_idx(il_first, xl_first + c)
                        patch_vel[0, :, c] = vel_data[idx][:depth_points]
                samples_vel.append(patch_vel)
                sample_il_list.append(trace_list[0][3] if velocity_per_trace else il_first)
                if save_positions:
                    pos = np.array([[t[3], t[4]] for t in trace_list], dtype=np.int32)
                    samples_positions.append(pos)
            total += 1

            if len(samples_seis) >= SAMPLES_PER_FILE:
                if len(samples_vel) != len(samples_seis):
                    raise RuntimeError("samples_vel len %d != samples_seis len %d" % (len(samples_vel), len(samples_seis)))
                seism = np.concatenate(samples_seis, axis=0)
                vmod = np.concatenate(samples_vel, axis=0)
                sp = os.path.join(out_dir, "seismic_full", "seismic%d.npy" % out_file_idx)
                vp = os.path.join(out_dir, "vmodel_full", "vmodel%d.npy" % out_file_idx)
                np.save(sp, seism)
                np.save(vp, vmod)
                if save_positions and samples_positions:
                    pos_arr = np.stack(samples_positions, axis=0)
                    pp = os.path.join(out_dir, "seismic_full", "sample_positions%d.npy" % out_file_idx)
                    np.save(pp, pos_arr)
                print("  Write %s %s, %s %s" % (sp, seism.shape, vp, vmod.shape))
                samples_seis, samples_vel = [], []
                if save_positions:
                    samples_positions = []
                out_file_idx += 1
    finally:
        for f in seismic_handles:
            f.close()

    if samples_seis:
        if len(samples_vel) != len(samples_seis):
            raise RuntimeError("Final batch: samples_vel len %d != samples_seis len %d" % (len(samples_vel), len(samples_seis)))
        seism = np.concatenate(samples_seis, axis=0)
        vmod = np.concatenate(samples_vel, axis=0)
        sp = os.path.join(out_dir, "seismic_full", "seismic%d.npy" % out_file_idx)
        vp = os.path.join(out_dir, "vmodel_full", "vmodel%d.npy" % out_file_idx)
        np.save(sp, seism)
        np.save(vp, vmod)
        if save_positions and samples_positions:
            pos_arr = np.stack(samples_positions, axis=0)
            pp = os.path.join(out_dir, "seismic_full", "sample_positions%d.npy" % out_file_idx)
            np.save(pp, pos_arr)
        print("  Write %s %s, %s %s" % (sp, seism.shape, vp, vmod.shape))

    inlines_arr = np.array(sample_il_list, dtype=np.int32)
    il_path = os.path.join(out_dir, "sample_inline.npy")
    np.save(il_path, inlines_arr)

    unique_il = np.unique(inlines_arr)
    n_il = len(unique_il)
    if n_il >= 3:
        n_train_il = int(round(n_il * 0.8))
        n_val_il = int(round(n_il * 0.1))
        n_test_il = n_il - n_train_il - n_val_il
        if n_test_il < 1:
            n_test_il = 1
            n_train_il = n_il - n_val_il - n_test_il
        il_to_split = {}
        for idx, il in enumerate(unique_il):
            if idx < n_train_il:
                il_to_split[il] = 0
            elif idx < n_train_il + n_val_il:
                il_to_split[il] = 1
            else:
                il_to_split[il] = 2
        split_arr = np.array([il_to_split[il] for il in inlines_arr], dtype=np.int32)
        split_msg = "8:1:1 by inline (shot il)"
    else:
        n = len(inlines_arr)
        n_train = max(1, int(round(n * 0.8)))
        n_test = max(1, int(round(n * 0.1)))
        n_val = n - n_train - n_test
        if n_val < 1:
            n_val = 1
            n_train = n - n_val - n_test
        perm = np.random.default_rng(42).permutation(n)
        split_arr = np.zeros(n, dtype=np.int32)
        split_arr[perm[:n_train]] = 0
        split_arr[perm[n_train : n_train + n_val]] = 1
        split_arr[perm[n_train + n_val :]] = 2
        split_msg = "8:1:1 by sample"
    split_path = os.path.join(out_dir, "split_by_inline.npy")
    np.save(split_path, split_arr)
    n_train = int((split_arr == 0).sum())
    n_val = int((split_arr == 1).sum())
    n_test = int((split_arr == 2).sum())
    # 写入元数据，便于后续判断是连续窗还是逐道对应
    meta_path = os.path.join(out_dir, "dataset_meta.txt")
    in_chans = 5 if (five_channel_mode or offset_sectored_mode) else 1
    with open(meta_path, "w", encoding="utf-8") as mf:
        mf.write("script=make_shot_gather_dataset.py\n")
        mf.write("velocity_mode=%s\n" % ("per_trace" if velocity_per_trace else "continuous_window"))
        mf.write("time_points=%d\n" % time_points)
        mf.write("depth_points=%d\n" % depth_points)
        mf.write("depth_range_m=%s\n" % (str(depth_range_m) if depth_range_m is not None else "truncate"))
        mf.write("n_crossline=%d\n" % n_cross)
        mf.write("downsample=%s\n" % str(downsample))
        mf.write("in_chans=%d\n" % in_chans)
        mf.write("channel_mode=%s\n" % ("five_shot" if five_channel_mode else ("offset_sectored" if offset_sectored_mode else "single")))
    print("Done. Total samples: %d. %s. train=%d val=%d test=%d" % (total, split_msg, n_train, n_val, n_test))
    print("Seismic shape per sample: (%d, %d, %d). Velocity: (1, %d, %d)." % (in_chans, time_points, n_cross, depth_points, n_cross))
    print("Metadata: %s (velocity_mode=%s)" % (meta_path, "per_trace" if velocity_per_trace else "continuous_window"))

    # 补零比例检查（allow_partial 时）
    if (five_channel_mode or offset_sectored_mode) and allow_partial and total > 0:
        sp1 = os.path.join(out_dir, "seismic_full", "seismic1.npy")
        if os.path.isfile(sp1):
            s1 = np.load(sp1)
            nz = (s1 != 0).sum()
            tot = s1.size
            print("[Zero-pad] Non-zero ratio: %.1f%% (if <10%% consider reducing overlap or check geometry)" % (100.0 * nz / tot))

    return total


def main():
    p = argparse.ArgumentParser(description="Shot gather dataset from seismic SEG-Y (no CMP), single or multi offset files")
    p.add_argument("--seismic", default=None, help="单地震 SEG-Y 文件（与 --seismic-files/--seismic-dir 二选一）")
    p.add_argument("--seismic-files", nargs="+", default=None, help="多 offset SEG-Y 路径列表")
    p.add_argument("--seismic-dir", default=None, help="地震目录，自动用该目录下 OFFSET_NEW_*.segy（如 <SEISMIC_DIR>）")
    p.add_argument("--limit-files", type=int, default=None, help="仅用前 N 个文件（逻辑验证用）")
    p.add_argument("--allow-partial", action="store_true", help="5炮模式：邻炮缺失道填0，不跳过（Rolling 几何时提高样本量）")
    p.add_argument("--min-xl-coverage", type=float, default=0.7, metavar="R", help="XL 方向最低覆盖率 0~1，如 0.7=允许缺30%%道补零；1.0=必须256道全有。默认 0.7")
    p.add_argument("--save-positions", action="store_true", help="保存每样本的 (il,xl) 到 sample_positions*.npy，用于对齐验证")
    p.add_argument("--velocity", required=True, help="速度 SEG-Y 文件")
    p.add_argument("--out", default="running/TrainDatasets_shot", help="输出目录")
    p.add_argument("--shot-by", choices=["fldr", "sx_sy", "ep_cdp"], default="fldr",
                   help="炮号字段: fldr(9-12) / sx_sy(73-80) / ep_cdp(17,21)；il/xl 仅做速度对齐")
    p.add_argument("--il", nargs=2, type=int, default=[INLINE_MIN, INLINE_MAX], metavar=("MIN", "MAX"))
    p.add_argument("--xl", nargs=2, type=int, default=[CROSSLINE_MIN, CROSSLINE_MAX], metavar=("MIN", "MAX"))
    p.add_argument("--max-samples", type=int, default=None, help="最多生成样本数")
    p.add_argument("--count-only", action="store_true", help="只统计有效炮数，不写文件")
    p.add_argument("--fixed-offset-targets", action="store_true", help="70 道按 70 个目标 offset 取最近道，使不同炮的列 c 对应同一 offset（参考 OpenFWI 固定几何）")
    p.add_argument("--offset-list", default=None, metavar="FILE", help="固定 offset 时从文件读 70 个整数（一行一个，可含 # 注释），如 running/offset_70_near_mid_far.txt")
    p.add_argument("--offset-min", type=int, default=0, help="固定偏移距时目标最小值（m），未用 --offset-list 时有效")
    p.add_argument("--offset-max", type=int, default=5175, help="固定偏移距时目标最大值（m），未用 --offset-list 时有效")
    p.add_argument("--time-points", type=int, default=3000, help="每道输出时间采样点数，默认 3000 (6s)；复现论文保留全波形")
    p.add_argument("--downsample", action="store_true", help="当 --time-points 小于原始道长时：True=从长序列均匀下采样得到 T 点；False=只取前 T 点（截断）")
    p.add_argument("--depth-points", type=int, default=256, help="速度输出深度点数，默认 256")
    p.add_argument("--depth-range", type=int, default=None, metavar="M", help="深度范围(米)，如 10000=10km；指定后从原始全深度重采样")
    p.add_argument("--depth-interval", type=float, default=10, metavar="M", dest="depth_interval_m", help="速度文件深度采样间隔(米)，默认10；若非10m请用 SeisView 确认后指定")
    p.add_argument("--skip-shot-gap", type=int, default=None, metavar="N", help="5炮模式：邻炮 fldr 差>N 时跳过(换线断点)，如 5")
    p.add_argument("--debug-five-shot", action="store_true", help="5炮模式：打印前5样本的 group 及各通道 hit 数，用于诊断")
    p.add_argument("--debug-discard", action="store_true", help="5炮模式：统计并打印样本被丢弃原因（0窗 vs all_have失败），诊断486瓶颈")
    p.add_argument("--receiver-by", choices=["ep_cdp", "gx_gy"], default="ep_cdp",
                   help="5炮模式：跨炮匹配用 ep_cdp(17,21) 或 gx_gy(81,85) 检波器坐标；若邻炮全黑可试 gx_gy")
    p.add_argument("--flip-flop", action="store_true",
                   help="5炮模式：奇偶震源，取 i-4,i-2,i,i+2,i+4（步长2）；先运行 check_source_geometry.py 确认")
    p.add_argument("--velocity-per-trace", action="store_true", default=True, help="[默认开启] 速度按每道 (il,xl) 抽取：地震第 c 列=检波器 c=速度第 c 列正下方，FWI 标准")
    p.add_argument("--no-velocity-per-trace", action="store_false", dest="velocity_per_trace", help="关闭逐道对应，改用第一道 (il,xl) 取连续窗")
    p.add_argument("--five-channels", action="store_true", help="5 炮模式 (OpenFWI)：5 个连续炮作为 5 通道，中间炮对齐，适配 ABA-FWI/DD-Net/AMFMS")
    p.add_argument("--offset-sectored", action="store_true", help="5 Offset 分通道（与 --five-channels 互斥）；5 通道=5 个 offset 段")
    p.add_argument("--n-crossline", type=int, default=None, help="检波器道数，默认 256（与论文一致）")
    p.add_argument("--sliding-window", action="store_true", help="2.5D 全工区滑窗：每炮切多窗，样本量=炮数*窗数")
    p.add_argument("--xl-starts", type=str, default=None, help="滑窗时 xl 起点，逗号分隔，如 1200,1600,2000,2400,2800")
    p.add_argument("--shot-decimation", type=int, default=1, help="滑窗/5炮时炮抽稀：每 N 炮取 1 炮，默认 1 不抽稀")
    args = p.parse_args()
    paths = None
    if args.seismic_dir:
        import glob
        paths = sorted(glob.glob(os.path.join(args.seismic_dir, "OFFSET_NEW_*.segy")))
        if not paths:
            p.error("--seismic-dir %s 下无 OFFSET_NEW_*.segy" % args.seismic_dir)
        if args.limit_files is not None:
            paths = paths[: args.limit_files]
            print("[limit-files] Using first %d files for logic test" % args.limit_files)
    elif args.seismic_files:
        paths = args.seismic_files
    elif args.seismic:
        paths = [args.seismic]
    if paths is None:
        p.error("需要 --seismic 或 --seismic-files 或 --seismic-dir")
    xl_starts = None
    if args.xl_starts:
        xl_starts = [int(x.strip()) for x in args.xl_starts.split(",")]
    run(seismic_path=paths[0] if len(paths) == 1 and args.seismic else None, seismic_paths=paths if len(paths) > 1 or args.seismic_files or args.seismic_dir else None, velocity_path=args.velocity, out_dir=args.out,
        shot_by=args.shot_by, il_range=tuple(args.il), xl_range=tuple(args.xl), max_samples=args.max_samples, count_only=args.count_only,
        fixed_offset_targets=args.fixed_offset_targets, offset_min=args.offset_min, offset_max=args.offset_max, offset_list_path=args.offset_list,
        time_points=args.time_points, downsample=args.downsample, depth_points=args.depth_points, depth_range_m=args.depth_range, depth_interval_m=args.depth_interval_m, velocity_per_trace=args.velocity_per_trace,
        five_channels=args.five_channels, sliding_window=args.sliding_window, xl_starts=xl_starts, shot_decimation=args.shot_decimation,
        offset_sectored=args.offset_sectored, n_crossline=args.n_crossline, allow_partial=args.allow_partial, min_xl_coverage=args.min_xl_coverage, save_positions=args.save_positions, skip_shot_gap=args.skip_shot_gap, debug_five_shot=args.debug_five_shot, debug_discard=args.debug_discard, receiver_by=args.receiver_by, flip_flop=args.flip_flop)


if __name__ == "__main__":
    main()
