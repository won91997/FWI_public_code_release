# Baseline References and Code Sources

This file records the paper citation and public/source-code link status for the
ten DL-FWI architectures evaluated in the paper. Public repository availability
and code-source status are recorded separately for clarity.

| Paper-facing model | Citation key in paper | Paper / source publication | DOI / source link | Code source used in this release | Code/source status |
| --- | --- | --- | --- | --- | --- |
| InversionNet | `Lin2022_InversionNet`; original method also `WuLin2020InversionNet` | Lin et al., *IEEE TGRS*, 2022; Wu and Lin, *IEEE TCI*, 2020 | `10.1109/TGRS.2021.3109011`; `10.1109/TCI.2019.2956866` | `OpenFWI/` and adapted training utilities; OpenFWI repository: `https://github.com/lanl/OpenFWI` | Upstream license included. |
| FCNVMB | `Yang2019_FCNVMB` | Yang and Ma, *Geophysics*, 2019 | `10.1190/geo2018-0249.1` | FCNVMB implementation through `ABA-FWI/` and `ddnet/` baseline wrappers | Author permission confirmed. |
| TU-Net | `WangXu2026TU-Net` | Wang et al., *Computers & Geosciences*, 2026 | `10.1016/j.cageo.2025.106028` | `TU-Net/`; upstream: `https://github.com/fansmale/TU-Net` | Author permission confirmed. |
| VIFNet | `Deng2025_VIFNet` | Deng et al., *Computers & Geosciences*, 2025 | `10.1016/j.cageo.2024.105834` | `VIF-Net/` local source; upstream: `https://github.com/FanSmale/VIF-dev` | Author permission confirmed. |
| DDNet70 | `Zhang2024_DDNet` | Zhang et al., *IEEE TGRS*, 2024 | `10.1109/TGRS.2024.3358492` | `ddnet/`; upstream: `https://github.com/FanSmale/ddnet` | Author permission confirmed. |
| DCNet | `Fu2025_DCNet` | Fu et al., *Journal of Applied Geophysics*, 2025 | `10.1016/j.jappgeo.2025.105762` | `DCNet/`; upstream: `https://github.com/FanSmale/DCNet` | Author permission confirmed. |
| ABA-FWI | `Xu_ABANet` | Xu et al., *IEEE TGRS*, 2024 | `10.1109/TGRS.2024.3496854` | `ABA-FWI/`; upstream: `https://github.com/FanSmale/ABA-FWI` | Author permission confirmed. |
| FuteFWI | `Li2025_CompGeo` | Li et al., *Computational Geosciences*, 2025 | `10.1007/s10596-025-10398-y` | `FuTE-FWI/`; upstream: `https://github.com/palemoons/FuTE-FWI` | Author permission confirmed. |
| ConvNeXt-FWI | `Kaggle2024_FWI` | Brendan Artley, Kaggle notebook for the Yale/UNC-Chapel Hill Geophysical Waveform Inversion competition, 2024 | Notebook: `https://www.kaggle.com/code/brendanartley/convnext-full-resolution-baseline`; competition: `https://www.kaggle.com/competitions/waveform-inversion` | `ConvNeXt-Kaggle/` legacy code path, paper-facing name `ConvNeXt-FWI` | Adapted source retained with citation to the public Kaggle notebook; pretrained weights are not redistributed. |
| VelocityGAN | `Zhang2019_VelocityGAN` | Zhang et al., *IEEE WACV*, 2019 | `10.1109/WACV.2019.00080` | Implemented through `FuTE-FWI/` and related baseline code | Author permission confirmed. |

## Practical Release Rule

For each baseline, keep three records:

1. The paper citation used in the manuscript.
2. The code source used in the benchmark, including upstream repository URL when available.
3. The code/source status used in this release.

This release records paper citations, source links, and code/source status for all compared baselines.
