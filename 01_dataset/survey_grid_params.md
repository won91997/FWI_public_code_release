# 工区网格参数（真实观测系统）

| 参数 | 值 | 说明 |
|------|-----|------|
| Inline 间距 | 25 m | 一个 inline 对应 25 m |
| Crossline 间距 | 12.5 m | 一个 crossline 对应 12.5 m |
| 深度采样 | 10 m | 深度方向每点 10 m |

上述参数已写入 `draw_observation_system.py` 常量：`INLINE_SPACING_M`、`CROSSLINE_SPACING_M`、`DEPTH_SAMPLE_M`。  
从 SEGY 道头读取 (il, xl) 后，会按 25 m / 12.5 m 换算为物理距离用于 3D 观测系统图。
