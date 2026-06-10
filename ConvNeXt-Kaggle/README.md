# ConvNeXt-FWI Source Reference

This directory contains the ConvNeXt-FWI implementation used in the paper. The
implementation was adapted from the following publicly available Kaggle
notebook:

- Brendan Artley, **ConvNeXt - Full Resolution Baseline**:
  https://www.kaggle.com/code/brendanartley/convnext-full-resolution-baseline
- Competition page: https://www.kaggle.com/competitions/waveform-inversion

The notebook page is publicly accessible on Kaggle and is cited here as the
source of this baseline. No pretrained weights are included in this release.

## Interface

The public benchmark imports `KaggleConvNeXtBaseline` from `kaggle_model.py`:


```python
class KaggleConvNeXtBaseline(torch.nn.Module):
    def __init__(
        self,
        output_size=(256, 256),
        pretrained=True,
        backbone="convnext_small.fb_in22k_ft_in1k",
        vmin=4158.69,
        vmax=6493.65,
    ):
        ...

    def forward(self, x):
        ...
```

The benchmark loader uses the model key `ConvNeXtKaggle`. Pretrained weights
are not included in this release.
