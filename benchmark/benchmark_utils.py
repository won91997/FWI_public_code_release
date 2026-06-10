"""Benchmark 通用工具：指标格式化等"""
import math


def fmt_sig(x, n=4):
    """将数值格式化为至少 n 位有效数字，便于区分模型差异。
    例：0.969 -> 0.9690, 27311.5 -> 27310, 0.0303 -> 0.03030
    """
    if x is None or (isinstance(x, str) and str(x).strip().upper() == "NA"):
        return "NA"
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if x == 0 or not math.isfinite(x):
        return str(x)
    if 0 < abs(x) < 1:
        exp = math.ceil(-math.log10(abs(x)))
        dec = n - 1 + exp
        return f"{x:.{dec}f}"
    else:
        exp = math.floor(math.log10(abs(x)))
        dec = max(0, n - 1 - exp)
        rounded = round(x, -exp + n - 1)
        return f"{rounded:.{dec}f}"
