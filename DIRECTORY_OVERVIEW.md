# Directory Overview

## Root Files

| File | Description |
| --- | --- |
| `README.md` | Public-facing usage guide for the release. |
| `PUBLIC_RELEASE_CONTENTS.md` | Curation record: what was kept, excluded, and what to configure locally. |
| `DIRECTORY_OVERVIEW.md` | This detailed directory inventory. |
| `THIRD_PARTY_NOTICES.md` | Third-party baseline source, permission status, and citation guidance. |
| `BASELINE_REFERENCES.md` | Paper citation, code source, and redistribution status for each evaluated model. |
| `DATA_AVAILABILITY.md` | Dataset availability, exclusions, and local reproduction format. |
| `CITATION.cff` | Citation metadata for the associated paper and public code package. |
| `MODEL_CODE_MAPPING.md` | Paper-name to code-key mapping for the ten evaluated models. |
| `LICENSE` | Root license notice. |
| `.gitignore` | Ignore rules for generated files, checkpoints, logs, and local config. |
| `config.example.env` | Template config for local reproduction. |
| `requirements.txt` | Root dependency pointer to the pinned benchmark Docker requirements. |
| `run_all.sh` | Full pipeline wrapper. |
| `run_resolution_constrained.sh` | Resolution-Constrained benchmark runner. |
| `run_full_resolution_nonfair.sh` | Target-Resolution benchmark runner. |
| `run_benchmark_suite.sh` | Main benchmark-suite launcher. |
| `run_benchmark_docker.sh` | Docker entry point for split preparation and paper-facing benchmark runs. |
| `profile_model_metrics.py` | Model parameter/FLOPs/inference profiling helper. |
| `scripts_run_*.sh` | Preserved historical/alternate launch scripts. |

## Workflow Directories

| Directory | Contents |
| --- | --- |
| `01_dataset/` | Dataset generation, five-shot preview, quality analysis, and survey parameter notes. |
| `02_prepare/` | Shared split/statistics preparation wrapper. |
| `03_train/` | Resolution-Constrained and Target-Resolution ten-model training entry scripts. |
| `04_eval/` | Evaluation refresh, SSIM rerun, single-checkpoint test, and result visualization scripts. |
| `benchmark/` | Unified benchmark implementation, evaluation metrics, model wrappers, split/stat scripts, Docker files, and implementation notes. |
| `results/` | Paper-facing CSV/Markdown result summaries and selected spectrum/preview figures. |

## Model/Baseline Directories

| Directory | Role |
| --- | --- |
| `OpenFWI/` | OpenFWI/InversionNet source, dataset utilities, transforms, visualization, scheduler, and split-file references. |
| `FuTE-FWI/` | FuTE-FWI, VelocityGAN, InversionNet training/test scripts, model definitions, and utilities. |
| `DCNet/` | DCNet model, training/testing scripts, configs, losses, and helper functions. |
| `ddnet/` | DDNet/FCNVMB/InversionNet source, sample README/figure assets, and model/result README references. |
| `TU-Net/` | TU-Net model definitions, training/testing scripts, configs, and data helpers. |
| `ABA-FWI/` | ABA-FWI source tree, including network definitions, data helpers, wavelet convolution utilities, and result scripts. |
| `VIF-Net/` | VIFNet model definition. |
| `ConvNeXt-Kaggle/` | ConvNeXt-FWI implementation adapted from the public Kaggle notebook; pretrained weights are not redistributed. |

## Removed From This Public Folder

| Removed Item | Reason |
| --- | --- |
| `*/.git/` | Nested repository metadata should not be published inside the release package. |
| `__pycache__/`, `*.pyc` | Python runtime cache. |
| `.DS_Store` | macOS metadata. |
| `core.*` | Local crash dumps. |
| `*.pth`, `*.pt`, `*.safetensors` | Checkpoints and pretrained weights are large binary artifacts. |
| `*.zip`, `*.z01`, `*.z02`, `*.z03`, `*.z04`, `*.rar` | Compressed data/model archives. |
| `logs_fair/`, `logs_nonfair/` | Local training logs. |
| `benchmark/generated_split/` | Generated split files contained machine-specific absolute paths. |

## Local Reproduction Checklist

- Confirm third-party model licenses and citations in `THIRD_PARTY_NOTICES.md`.
- Copy `config.example.env` to `config.env` for local runs only; do not commit `config.env`.
- Regenerate `benchmark/generated_split/` from your local `DATA_ROOT`.
- Paper experiments use fixed random seed `42` by default.
- For Docker, mount the repository to `/workspace/repo` and the dataset to `/workspace/data`.
