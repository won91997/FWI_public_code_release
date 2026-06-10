# FWI Ten-Model Benchmark Pipeline

This repository contains the public code package for the FAN-10000m DL-FWI
benchmark workflow. It covers dataset preparation, shared train/validation/test
split generation, ten-model training, unified evaluation, model profiling, and
result summary generation.

The FAN-10000m field data are not included. The original working directory also
contained local logs, checkpoints, pretrained weights, cache files, nested Git
metadata, and machine-specific paths; those files are intentionally excluded
from this public release folder.

## What Is Included

| Path | Purpose |
| --- | --- |
| `01_dataset/` | Build OpenFWI-style 256-trace, five-shot datasets and inspect data quality. |
| `02_prepare/` | Generate shared split files and training statistics. |
| `03_train/` | Entry scripts for Resolution-Constrained and Target-Resolution ten-model training. |
| `04_eval/` | Single-checkpoint testing, SSIM rerun, figure generation, and summary refresh. |
| `benchmark/` | Core unified benchmark framework: loaders, wrappers, training, testing, metrics, Docker setup. |
| `OpenFWI/` | OpenFWI/InversionNet-related baseline code. |
| `FuTE-FWI/` | FuTE-FWI, VelocityGAN, and related baseline code. |
| `DCNet/` | DCNet baseline code. |
| `ddnet/` | DDNet/FCNVMB-related baseline code kept from the comparison workspace. |
| `TU-Net/` | TU-Net baseline code. |
| `ABA-FWI/` | ABA-FWI baseline code. |
| `VIF-Net/` | VIFNet model definition. |
| `ConvNeXt-Kaggle/` | ConvNeXt-FWI implementation adapted from the public Kaggle notebook. Pretrained weights are not included. |
| `results/` | Paper-facing result tables and selected figures. |
| `BASELINE_REFERENCES.md` | Paper citation, code source, and code/source status for each evaluated model. |
| `THIRD_PARTY_NOTICES.md` | Third-party baseline sources, permissions, and citation guidance. |
| `DATA_AVAILABILITY.md` | Dataset availability boundaries and local reproduction format. |

## Evaluated Models and Code Locations

The paper reports ten architectures. Some code identifiers keep legacy names from
the original comparison workspace; the paper-facing names are listed below.

| Paper name | Code key / script name | Main location | Note |
| --- | --- | --- | --- |
| InversionNet | `InversionNet` | `OpenFWI/`, `FuTE-FWI/` | Native training script with unified loader support. |
| FCNVMB | `FCNVMB` / `FCNVMB_FWI` | `ABA-FWI/`, `ddnet/` | FCN baseline wrapper used by the unified benchmark. |
| TU-Net | `TU_Net` | `TU-Net/` | U-Net-style local-global baseline. |
| VIFNet | `VIFNet` | `VIF-Net/` | Run by the top-level wrappers outside the main-eight suite. |
| DDNet70 | `DDNet70` | `ddnet/` | Dual-decoder baseline. |
| DCNet | `DCNet` | `DCNet/` | Difference-convolution baseline. |
| ABA-FWI | `ABA_FWI` | `ABA-FWI/` | Boundary-aware baseline. |
| FuteFWI | `FuteFWI` | `FuTE-FWI/` | Transformer baseline. |
| ConvNeXt-FWI | `ConvNeXtKaggle` | `ConvNeXt-Kaggle/` | Adapted from the public Kaggle notebook by Brendan Artley; pretrained weights are not included. |
| VelocityGAN | `VelocityGAN` | `FuTE-FWI/` | GAN baseline. |

## Main Scripts

| Script | Purpose |
| --- | --- |
| `run_all.sh` | Optional wrapper for the full pipeline. |
| `run_resolution_constrained.sh` | Resolution-Constrained benchmark runner. It uses `BENCHMARK_FAIR_MODE=1` internally for backward compatibility. |
| `run_full_resolution_nonfair.sh` | Target-Resolution benchmark runner. It uses `BENCHMARK_FAIR_MODE=0` internally for backward compatibility. |
| `run_benchmark_suite.sh` | Lower-level runner for the eight shared-task models. ConvNeXt-FWI and VIFNet are added by the top-level wrappers. |
| `run_benchmark_docker.sh` | Docker entry point for split preparation plus paper-facing benchmark runs. |
| `03_train/run_fair_10models.sh` | Train ten models under the Resolution-Constrained protocol. |
| `03_train/run_nonfair_10models.sh` | Train ten models under the Target-Resolution protocol. |
| `04_eval/run_refresh_summaries.sh` | Refresh result summaries from evaluation outputs. |

## External Files Not Included

The following files were excluded because they are generated artifacts, local
machine state, or large binary assets:

- model checkpoints: `*.pth`, `*.pt`
- pretrained model weights: `*.safetensors`
- compressed model/data archives: `*.zip`, `*.z01`, `*.z02`, `*.z03`, `*.z04`, `*.rar`
- Python caches: `__pycache__/`, `*.pyc`
- macOS metadata: `.DS_Store`
- crash dumps: `core.*`
- local logs: `logs_fair/`, `logs_nonfair/`
- nested Git metadata: `*/.git/`
- generated split files with absolute local paths: `benchmark/generated_split/`

## Dataset Layout

Training scripts expect an OpenFWI-style dataset root:

```text
DATA_ROOT/
├── seismic_full/
│   ├── seismic1.npy
│   ├── seismic2.npy
│   └── ...
└── vmodel_full/
    ├── vmodel1.npy
    ├── vmodel2.npy
    └── ...
```

Expected shapes:

- seismic: `(N, 5, 3000, 256)`
- velocity model: `(N, 256, 256)`

If only SEG-Y resources are available, see `01_dataset/run_make_dataset.sh` and
set the corresponding private paths in your local `config.env`. Do not commit
raw SEG-Y files or generated arrays to this repository.

## Quick Start

```bash
cp config.example.env config.env
source config.env

# Generate shared split and train statistics.
bash 02_prepare/run_prepare.sh

# Resolution-Constrained benchmark.
bash 03_train/run_fair_10models.sh

# Target-Resolution benchmark.
bash 03_train/run_nonfair_10models.sh

# Refresh result summaries.
bash 04_eval/run_refresh_summaries.sh
```

For the Docker workflow and pinned Python dependencies, see
`benchmark/docker/BUILD_AND_RUN.md` and `benchmark/docker/requirements.txt`.
The same dependency list is exposed at repository root through
`requirements.txt`.

## Reproducibility Notes

- Default training assumes multi-GPU CUDA execution through `torchrun`.
- Paper experiments use fixed random seed `42`. Set `GPU_IDS`, `SEEDS`, `BATCH_PER_GPU`, and `EPOCHS` in `config.env`.
- The benchmark split is generated locally from `DATA_ROOT`; do not reuse split
  CSV files generated on another machine unless paths have been normalized.
- ConvNeXt-FWI pretrained weights are not shipped in this release. The source
  implementation is retained with citation to the public Kaggle notebook.
- Third-party baseline folders are retained for reproducibility. Please cite the
  original projects where required.

## Publication Notes

- Review `THIRD_PARTY_NOTICES.md` and `BASELINE_REFERENCES.md` for third-party
  baseline sources, permissions, and citation requirements.
- ConvNeXt-FWI is traced to Brendan Artley's public Kaggle notebook
  (`https://www.kaggle.com/code/brendanartley/convnext-full-resolution-baseline`).
  The adapted source is retained for reproducibility, while pretrained weights
  are not included.
- Historical helper scripts and upstream split files may contain placeholder
  paths such as `<DATA_ROOT>` or upstream dataset paths. Main public entry
  scripts read paths from `config.env` or environment variables.
- Do not commit `config.env`, training logs, checkpoints, raw SEG-Y files, or
  generated `.npy` arrays.
