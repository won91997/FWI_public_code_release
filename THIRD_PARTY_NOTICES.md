# Third-Party Notices

This release includes code adapted from or copied from multiple public FWI
model repositories for reproducible benchmarking. Known upstream sources are listed below.

| Local path | Upstream project | Notes |
| --- | --- | --- |
| `OpenFWI/` | `https://github.com/lanl/OpenFWI` | Upstream license included. |
| `FuTE-FWI/` | `https://github.com/palemoons/FuTE-FWI` | Author permission confirmed. |
| `DCNet/` | `https://github.com/FanSmale/DCNet` | Author permission confirmed. |
| `ddnet/` | `https://github.com/FanSmale/ddnet` | Author permission confirmed. Cite Zhang et al. (TGRS 2024, DOI: 10.1109/TGRS.2024.3358492). |
| `TU-Net/` | `https://github.com/fansmale/TU-Net` | Author permission confirmed. |
| `ABA-FWI/` | `https://github.com/FanSmale/ABA-FWI` | Author permission confirmed. |
| `VIF-Net/` | `https://github.com/FanSmale/VIF-dev` | Author permission confirmed. |
| `ConvNeXt-Kaggle/` (paper name: ConvNeXt-FWI; legacy code path retained) | `https://www.kaggle.com/code/brendanartley/convnext-full-resolution-baseline` | Source is recorded as Brendan Artley's public Kaggle notebook, with the competition page at `https://www.kaggle.com/competitions/waveform-inversion`. The adapted source is retained for reproducibility; pretrained weights are not included. |


## Confirmed Public Code Links

- DD-Net / DDNet70: `https://github.com/FanSmale/ddnet`. Author permission confirmed.

## Citation Guidance

For model-by-model paper citations, DOI information, and code-source mapping, see `BASELINE_REFERENCES.md`.

When publishing this repository, cite the original papers and repositories for
all baseline models, in addition to the paper associated with this benchmark
pipeline.

## Known Upstream Warnings

- `ABA-FWI/ABA-FWI_2.0/marmousi/joint_slice.py` contains a Windows-style string
  that triggers a Python invalid-escape `SyntaxWarning` during syntax checks.
  This is retained as upstream baseline code and is not part of the main public
  benchmark entry path.

## License Status

The root `LICENSE` is a conservative all-rights-reserved notice. Replace it with
a standard open-source license only after the authors decide how to license
their own benchmark orchestration code.
