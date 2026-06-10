# Data Availability

The full FAN-10000m dataset used in the paper is not included in this code
release.

## What Is Included

- Code for generating an OpenFWI-style dataset from local velocity/seismic
  resources.
- Unified train/validation/test split generation scripts.
- Dataset quality analysis and visualization scripts.
- Result tables and selected figures used for paper-facing summaries.

## What Is Not Included

- Raw SEG-Y files.
- Full generated seismic/velocity `.npy` arrays.
- Model checkpoints and pretrained weights.
- Generated split CSV files containing machine-specific absolute paths.

## Expected Dataset Format

The benchmark expects a dataset root with the following structure:

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

Expected array shapes:

- seismic: `(N, 5, 3000, 256)`
- velocity model: `(N, 256, 256)`

## Using Your Own Data

1. Prepare data in the expected `seismic_full/` and `vmodel_full/` layout.
2. Set `DATA_ROOT` in `config.example.env`, then copy it to `config.env` for a
   private local run.
3. Run `bash 02_prepare/run_prepare.sh` to generate split files and training
   statistics for your local dataset path.
4. Run either the Resolution-Constrained or Target-Resolution comparison scripts.

Do not commit generated split files if they contain local absolute paths.
