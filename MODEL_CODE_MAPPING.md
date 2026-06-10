# Model Code Mapping

The paper reports ten DL-FWI architectures. This file maps the paper-facing
model names to the public release code identifiers.

| Paper name | Code key | Main folder | Included in `run_benchmark_suite.sh` | Included by top-level wrappers | Source note |
| --- | --- | --- | --- | --- | --- |
| InversionNet | `InversionNet` | `OpenFWI/`, `FuTE-FWI/` | yes | yes | OpenFWI-related baseline. |
| FCNVMB | `FCNVMB`, `FCNVMB_FWI` | `ABA-FWI/`, `ddnet/` | yes | yes | FCN baseline wrapper. |
| TU-Net | `TU_Net` | `TU-Net/` | yes | yes | See THIRD_PARTY_NOTICES.md. |
| VIFNet | `VIFNet` | `VIF-Net/` | no | yes | Upstream: `https://github.com/FanSmale/VIF-dev`; upstream repository identified from the published paper; author permission confirmed. |
| DDNet70 | `DDNet70` | `ddnet/` | yes | yes | Upstream: `https://github.com/FanSmale/ddnet`; cite Zhang et al. (TGRS 2024). |
| DCNet | `DCNet` | `DCNet/` | yes | yes | Upstream: `https://github.com/FanSmale/DCNet`; See THIRD_PARTY_NOTICES.md. |
| ABA-FWI | `ABA_FWI` | `ABA-FWI/` | yes | yes | Upstream: `https://github.com/FanSmale/ABA-FWI`; See THIRD_PARTY_NOTICES.md. |
| FuteFWI | `FuteFWI` | `FuTE-FWI/` | yes | yes | See THIRD_PARTY_NOTICES.md. |
| ConvNeXt-FWI | `ConvNeXtKaggle` | `ConvNeXt-Kaggle/` | yes | yes | Adapted from the public Kaggle notebook: `https://www.kaggle.com/code/brendanartley/convnext-full-resolution-baseline`; competition page: `https://www.kaggle.com/competitions/waveform-inversion`. |
| VelocityGAN | `VelocityGAN` | `FuTE-FWI/` | yes | yes | Implemented through FuTE-FWI baseline folder. |

Notes:

- `run_benchmark_suite.sh` is the lower-level shared runner for eight models.
- `run_resolution_constrained.sh` and `run_full_resolution_nonfair.sh` add
  ConvNeXt-FWI and VIFNet, yielding the ten-model benchmark used by the paper.
- `BENCHMARK_FAIR_MODE=1` is the legacy code flag for the paper's
  Resolution-Constrained setting.
- `BENCHMARK_FAIR_MODE=0` is the legacy code flag for the paper's
  Target-Resolution setting.
