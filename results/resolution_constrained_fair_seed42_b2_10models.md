# Resolution-Constrained 10 模型汇总

配置：`fair` + `seed42` + `batch=2`

## 说明

- 这 10 个模型来自你指定的名单。
- 这里只有一个 GAN：`VelocityGAN`。
- 如果把之前 benchmark 的 `ConvNeXtFWI` 也算进去，就会变成 11 个模型。
- `ConvNeXtKaggle` 与 `VIFNet` 使用单独目录补充进总表。

## 总表

| 模型 | Params(M) | FLOPs(G) | Train(h) | Infer(ms) | MSE | MAE | PSNR | SSIM | L1-Grad | LPIPS | 状态 | 来源 |
|------|-----------:|---------:|---------:|----------:|----:|----:|-----:|-----:|--------:|------:|------|------|
| ConvNeXtKaggle | 53.7973 | 5.60081 |  | 29.5978 | 21380.0 | 116.6 | 23.37 | 0.968 | 11.97 | 0.03186 | OK | `benchmark_logs_convnext_kaggle_seed42_b2` |
| FuteFWI | 44.3837 | 17.4718 | 3.39806 | 11.2308 | 25240 | 111.9 | 22.65 | 0.961 | 10.46 | 0.03837 | OK | `benchmark_logs_fair_nativeloss_seed42_all` |
| InversionNet | 15.103 | 1.44532 | 0.980556 | 3.28867 | 17180 | 96.67 | 24.32 | 0.9739 | 9.620 | 0.02571 | OK | `benchmark_logs_fair_nativeloss_seed42_all` |
| VelocityGAN | 15.1207 | 11.6802 | 4.29194 | 13.6481 | 17030 | 92.31 | 24.36 | 0.974 | 13.60 | 0.02562 | OK | `benchmark_logs_fair_nativeloss_seed42_all` |
| DCNet | 20.4461 | 1.80132 | 0.959722 | 4.18387 | 17430 | 95.19 | 24.26 | 0.9733 | 9.724 | 0.02627 | OK | `benchmark_logs_fair_nativeloss_seed42_all` |
| DDNet70 | 10.5432 | 1.71631 | 1.01278 | 3.49895 | 16730 | 94.03 | 24.44 | 0.9745 | 9.513 | 0.02498 | OK | `benchmark_logs_fair_nativeloss_seed42_all` |
| TU_Net | 10.1912 | 4.85813 | 1.21556 | 5.39435 | 16810 | 95.16 | 24.42 | 0.9746 | 9.387 | 0.02476 | OK | `benchmark_logs_fair_nativeloss_seed42_all` |
| ABA_FWI | 31.2012 | 5.1047 | 1.18361 | 5.90564 | 16530 | 95.98 | 24.49 | 0.9751 | 9.496 | 0.02464 | OK | `benchmark_logs_fair_nativeloss_seed42_all` |
| FCNVMB | 31.0447 | 637.16 | 1.135 | 230.967 | 15670 | 92.33 | 24.72 | 0.9761 | 9.401 | 0.02353 | OK | `benchmark_logs_fair_nativeloss_seed42_all` |
| VIFNet | 96.3208 | 25.8191 | 1.913611 | 19.3756 | 17230.0 | 97.42 | 24.31 | 0.9744 | 9.613 | 0.02519 | OK | `benchmark_logs_fair_nativeloss_vifnet_seed42` |

## 按 PSNR 排名

| 排名 | 模型 | PSNR | SSIM | LPIPS |
|------|------|-----:|-----:|------:|
| 1 | FCNVMB | 24.72 | 0.9761 | 0.02353 |
| 2 | ABA_FWI | 24.49 | 0.9751 | 0.02464 |
| 3 | DDNet70 | 24.44 | 0.9745 | 0.02498 |
| 4 | TU_Net | 24.42 | 0.9746 | 0.02476 |
| 5 | VelocityGAN | 24.36 | 0.974 | 0.02562 |
| 6 | InversionNet | 24.32 | 0.9739 | 0.02571 |
| 7 | VIFNet | 24.31 | 0.9744 | 0.02519 |
| 8 | DCNet | 24.26 | 0.9733 | 0.02627 |
| 9 | ConvNeXtKaggle | 23.37 | 0.968 | 0.03186 |
| 10 | FuteFWI | 22.65 | 0.961 | 0.03837 |
