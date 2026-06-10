# Public Release Contents

This file records how the public code folder was curated from the original
working directory.

## Kept

- pipeline entry scripts at repository root
- dataset creation and quality analysis scripts in `01_dataset/`
- split/statistics preparation script in `02_prepare/`
- Resolution-Constrained and Target-Resolution training launchers in `03_train/`
- evaluation and summary scripts in `04_eval/`
- benchmark framework in `benchmark/`
- baseline model source folders:
  - `OpenFWI/`
  - `FuTE-FWI/`
  - `DCNet/`
  - `ddnet/`
  - `TU-Net/`
  - `ABA-FWI/`
  - `VIF-Net/`
  - `ConvNeXt-Kaggle/` (paper name: ConvNeXt-FWI; adapted source retained, pretrained weights excluded)
- paper-facing result CSV/Markdown files and selected figures in `results/`
- publication support files: `DATA_AVAILABILITY.md`, `CITATION.cff`,
  `THIRD_PARTY_NOTICES.md`, `BASELINE_REFERENCES.md`, and `LICENSE`

## Excluded

- nested `.git/` directories from copied third-party repositories
- Python bytecode and cache directories
- macOS `.DS_Store` files
- crash dumps such as `OpenFWI/core.*`
- model checkpoints and pretrained weights
- compressed model/data archives
- local training logs
- generated split CSV files containing machine-specific absolute paths

## Important Follow-Up Before Publishing

1. Confirm all third-party baseline licenses and citations.
2. Copy `config.example.env` to `config.env` only for private local runs; do not
   commit machine-specific `config.env`.
3. Regenerate `benchmark/generated_split/` on the target machine from the local
   dataset path.
4. Paper experiments use fixed random seed `42` by default.
5. If pretrained weights are needed, publish them separately with checksums or
   document the download source.
