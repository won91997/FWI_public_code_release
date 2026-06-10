#!/usr/bin/env python3
import argparse
import os
import sys
import time
import torch


def _ensure_repo_path(repo):
    """Ensure repo root is in sys.path. Script runs from WORK_ROOT; imports need repo dir in path."""
    from pathlib import Path
    root = Path(__file__).resolve().parent
    path_map = {
        "openfwi": root / "OpenFWI",
        "futefwi": root / "FuTE-FWI",
        "dcnet": root / "DCNet",
        "ddnet": root / "ddnet",
        "tu-net": root / "TU-Net",
        "aba-fwi": root / "ABA-FWI" / "ABA-FWI_2.0",
        "convnext-kaggle": root / "ConvNeXt-Kaggle",
        "vif-net": root / "VIF-Net",
    }
    path = str(path_map.get(repo))
    if path and path not in sys.path:
        sys.path.insert(0, path)


def parse_args():
    p = argparse.ArgumentParser("Profile params/flops/infer for known FWI models")
    p.add_argument("--repo", choices=["openfwi", "futefwi", "dcnet", "ddnet", "tu-net", "aba-fwi", "convnext-kaggle", "vif-net"], required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--shape", default="1,5,3000,256", help="input shape as N,C,T,X")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    return p.parse_args()


def count_params_m(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def get_model(repo, model_name, output_size=(256, 256), benchmark_key=None):
    _ensure_repo_path(repo)
    if repo == "openfwi":
        import network  # type: ignore

        if model_name not in network.model_dict:
            raise ValueError(f"Unknown openfwi model: {model_name}")
        return network.model_dict[model_name](output_size=output_size)
    if repo == "futefwi":
        from models import FuteFWI, InversionNet, Generator  # type: ignore

        if model_name == "FuteFWI":
            return FuteFWI(output_size=output_size)
        if model_name == "InversionNet":
            return InversionNet(output_size=output_size)
        if model_name == "VelocityGAN":
            return Generator(output_size=output_size)
        raise ValueError(f"Unknown futefwi model: {model_name}")
    if repo == "dcnet":
        from DCNet import DCModel  # type: ignore
        from func.ddnet import DDNet70Model  # type: ignore
        from func.comparison_net import InversionNet  # type: ignore

        if model_name == "DCNet":
            return DCModel(output_size=output_size)
        if model_name == "DDNet70":
            return DDNet70Model(n_classes=1, in_channels=5, is_deconv=True, is_batchnorm=True, output_size=output_size)
        if model_name == "InversionNet":
            return InversionNet(output_size=output_size)
        raise ValueError(f"Unknown dcnet model: {model_name}")
    if repo == "ddnet":
        from net.DDNet70 import DDNet70Model, SDNet70Model  # type: ignore
        from net.InversionNet import InversionNet  # type: ignore
        from net.FCNVMB import FCNVMB  # type: ignore

        if model_name == "DDNet70":
            return DDNet70Model(n_classes=1, in_channels=5, is_deconv=True, is_batchnorm=True, output_size=output_size)
        if model_name == "SDNet70":
            return SDNet70Model(n_classes=1, in_channels=5, is_deconv=True, is_batchnorm=True, output_size=output_size)
        if model_name == "InversionNet":
            return InversionNet(output_size=output_size)
        if model_name == "FCNVMB":
            return FCNVMB(n_classes=1, in_channels=5, is_deconv=True, is_batchnorm=True)
        raise ValueError(f"Unknown ddnet model: {model_name}")
    if repo == "tu-net":
        from model.TU_Net import TU_Net  # type: ignore
        from model.DDNet70 import DDNet70Model, SDNet70Model  # type: ignore
        from model.InversionNet import InversionNet  # type: ignore
        from model.FCNVMB import FCNVMB  # type: ignore

        if model_name == "TU_Net":
            return TU_Net(n_classes=1, in_channels=5, is_deconv=True, is_batchnorm=True, output_size=output_size)
        if model_name == "DDNet70":
            return DDNet70Model(n_classes=1, in_channels=5, is_deconv=True, is_batchnorm=True, output_size=output_size)
        if model_name == "SDNet70":
            return SDNet70Model(n_classes=1, in_channels=5, is_deconv=True, is_batchnorm=True, output_size=output_size)
        if model_name == "InversionNet":
            return InversionNet(output_size=output_size)
        if model_name == "FCNVMB":
            return FCNVMB(n_classes=1, in_channels=5, is_deconv=True, is_batchnorm=True)
        raise ValueError(f"Unknown tu-net model: {model_name}")
    if repo == "aba-fwi":
        from net.ABA_FWI import ABA_FWI  # type: ignore
        from net.FCNVMB import FCNVMB_FWI  # type: ignore

        if model_name == "ABA_FWI":
            return ABA_FWI(output_size=output_size)
        if model_name == "ABA_Loss":
            # ABA_Loss 不在 benchmark 的 ABA_FWI.py 中，仅 ABA_FWI/FCNVMB_FWI 可 profile
            raise ValueError("ABA_Loss not in net.ABA_FWI; profile supports ABA_FWI, FCNVMB_FWI only")
        if model_name == "FCNVMB_FWI":
            return FCNVMB_FWI(model_dim=output_size, in_channels=5)
        raise ValueError(f"Unknown aba-fwi model: {model_name}")
    if repo == "convnext-kaggle":
        from kaggle_model import KaggleConvNeXtBaseline  # type: ignore

        if model_name == "ConvNeXtKaggle":
            use_pretrained = os.environ.get("KAGGLE_CONVNEXT_PRETRAINED", "1") == "1"
            backbone = os.environ.get("KAGGLE_CONVNEXT_BACKBONE", "convnext_small.fb_in22k_ft_in1k")
            return KaggleConvNeXtBaseline(output_size=output_size, pretrained=use_pretrained, backbone=backbone)
        raise ValueError(f"Unknown convnext-kaggle model: {model_name}")
    if repo == "vif-net":
        from VIFNet import VIFNet  # type: ignore

        if model_name == "VIFNet":
            return VIFNet(output_size=output_size, in_channels=5)
        raise ValueError(f"Unknown vif-net model: {model_name}")
    raise ValueError(f"Unsupported repo: {repo}")


def get_model_for_benchmark(repo, model_name, output_size=(256, 256), benchmark_key=None):
    """若 BENCHMARK_FAIR_MODE=1 则对配置内模型使用 FairWrapper。"""
    use_fair = os.environ.get("BENCHMARK_FAIR_MODE", "0") == "1"
    key = benchmark_key or model_name
    if use_fair and key in get_model_for_benchmark._FAIR_CONFIG:
        internal = get_model_for_benchmark._FAIR_CONFIG[key]
        model = get_model(repo, model_name, (internal, internal))
        from benchmark.fair_wrapper import FairWrapper
        return FairWrapper(model, target_size=output_size, internal_size=(internal, internal))
    return get_model(repo, model_name, output_size)


get_model_for_benchmark._FAIR_CONFIG = {
    "ABA_FWI": 70, "FCNVMB": 70, "FCNVMB_FWI": 70,  # suite 用 --model FCNVMB_FWI，需同时支持
    "TU_Net": 64, "DDNet70": 64, "DCNet": 70, "ConvNeXtKaggle": 70,
    "InversionNet": 70, "FuteFWI": 70, "VIFNet": 64,
}


def _model_forward(model, x, repo, model_name, output_size=(256, 256)):
    """Unified forward for profiling; some models need extra args (label_dsp_dim)."""
    label_dsp = [output_size[0], output_size[1]]
    if repo == "dcnet" and model_name == "DCNet":
        return model(x, label_dsp)
    if repo in ("ddnet", "tu-net") and model_name == "FCNVMB":
        return model(x, label_dsp)
    if repo == "vif-net" and model_name == "VIFNet":
        out = model(x)
        return out[0] if isinstance(out, list) else out
    return model(x)


class _ProfileWrapper(torch.nn.Module):
    """Wrapper so thop.profile sees a simple forward(x)."""

    def __init__(self, model, repo, model_name):
        super().__init__()
        self._model = model
        self._repo = repo
        self._name = model_name

    def forward(self, x):
        return _model_forward(self._model, x, self._repo, self._name)


def try_flops_g(model, x, repo, model_name):
    try:
        from thop import profile  # type: ignore

        wrapped = _ProfileWrapper(model, repo, model_name)
        flops, _ = profile(wrapped, inputs=(x,), verbose=False)
        return flops / 1e9
    except Exception:
        return None


def infer_ms(model, x, warmup, iters, use_cuda, repo, model_name):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = _model_forward(model, x, repo, model_name)
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = _model_forward(model, x, repo, model_name)
        if use_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / iters


def main():
    args = parse_args()
    shape = tuple(int(v) for v in args.shape.split(","))
    if len(shape) != 4:
        raise ValueError("--shape must be N,C,T,X")

    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = get_model_for_benchmark(args.repo, args.model, benchmark_key=args.model).to(device)
    x = torch.randn(*shape, device=device)

    params_m = count_params_m(model)
    # Run infer_ms BEFORE try_flops_g: thop.profile registers hooks that can corrupt the model
    # (e.g. ReLU total_ops AttributeError with shared modules in DDNet70/TU_Net). Running infer
    # first ensures we get params+infer for all models; flops may be NA if thop fails.
    infer = infer_ms(model, x, args.warmup, args.iters, use_cuda, args.repo, args.model)
    flops_g = try_flops_g(model, x, args.repo, args.model)

    # 至少 4 位有效数字，便于区分模型差异
    flops_str = "NA" if flops_g is None else f"{flops_g:.6g}"
    print(f"METRICS params_m={params_m:.6g} flops_g={flops_str} infer_ms={infer:.6g}")


if __name__ == "__main__":
    main()
