# CFD surrogate validation report

## Scope

This report summarizes the reproducible public-data experiment. It uses 32
independent `CFDBench/cylinder/bc` cases, a strict whole-case split, and
train-only normalization. The selected release contains interpolated velocity
fields `u` and `v`; pressure is not available.

| Split | Cases | Samples | Test grouping |
|---|---:|---:|---|
| Train | 22 | 880 | — |
| Validation | 5 | 200 | — |
| Test | 5 | 200 | 2 interpolation + 3 extrapolation cases |

The branch is `[Re, U_inlet, time_norm]`; the trunk is `[x/D, y/D]`. The
query grid is 48×32 for the selected best configuration. Points inside the
cylinder are excluded from loss and metrics.

## Ablation results

All rows use the same case-level test split and seed. The best row is selected
by full-field RMSE.

| Experiment | Grid | Parameters | Full RMSE | Relative L2 | Wake RMSE |
|---|---:|---:|---:|---:|---:|
| Baseline | 24×16 | 47,330 | 0.095425 | 0.094649 | 0.111985 |
| Larger MLP | 24×16 | 181,602 | 0.091521 | 0.090777 | 0.107904 |
| Fourier coordinates | 24×16 | 184,162 | 0.091458 | 0.090715 | 0.107728 |
| Best configuration | 48×32 | 184,162 | **0.091270** | **0.088385** | **0.106116** |

## Baseline comparison

All methods below use the same five test cases, time samples, and obstacle
mask.

| Method | Full RMSE | Wake RMSE |
|---|---:|---:|
| DeepONet | **0.091270** | **0.106116** |
| Training mean-field | 0.140311 | 0.160382 |
| Linear-Re/time interpolation | 0.207626 | 0.242343 |
| Uniform-inlet field | 0.765044 | 0.775069 |

The interpolation subset RMSE is `0.096940` for DeepONet and `0.118531` for
the linear-Re/time baseline. The extrapolation subset RMSE is `0.087286`.

The current public-data experiment therefore validates training, case-level
evaluation, and portability. It does not establish DeepONet as a universal or
production-best CFD surrogate. The results are for a 2D cylinder velocity
benchmark and should not be presented as monopile or pressure-model results.

## Reproducibility notes

- The source archive is accessed selectively; the full archive is not needed
  to inspect the repository code.
- Case splits are disjoint by construction, and normalization statistics are
  fitted from training cases only.
- The historical internal filenames contain `v2` to preserve script/config
  compatibility; the public project name is **DeepONet CFD Surrogate**.
