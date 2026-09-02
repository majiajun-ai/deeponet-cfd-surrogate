# Portability report

The same training/evaluation code was exercised across three runtime paths.
HPC identifiers are intentionally omitted; reproduce them with your own
`<cpu-node>`, `<ascend-node>`, and scheduler configuration.

| Target | Architecture | Runtime/backend | Smoke | Observation |
|---|---|---|---|---|
| Windows | amd64 | Docker CPU | PASS | Real HDF5 one-epoch smoke completed |
| Kunpeng CPU | arm64/aarch64 | Native user-space Python + Donau + CPU PyTorch | PASS | 7.5315 s/epoch; test-field RMSE matched the Windows smoke |
| Ascend 910B | arm64/aarch64 | Native Python + PyTorch + `torch_npu` + CANN + Donau | PASS | 250.0818 s/epoch; field RMSE relative difference `2.47e-5` |

The migration smoke uses the same 47,330-parameter configuration, seed, data
split, and real 24×16 HDF5 dataset. The evaluated test-field RMSE values were:

| Checkpoint | Test field RMSE |
|---|---:|
| Windows CPU baseline | 0.3517240882 |
| Kunpeng CPU | 0.3517240882 |
| Ascend checkpoint evaluated with the common CPU metric path | 0.3517327905 |

The small model is not a useful NPU performance workload: the measured wall
time is much higher on one Ascend 910B than on Kunpeng CPU. This is a
workload-specific hardware-fit conclusion, not a general statement about
Ascend performance.

## Route decisions

- **Recommended portability target:** Kunpeng ARM64 CPU with native user-space
  Python under Donau.
- **Ascend status:** single-NPU compatibility path is runnable; it is not an
  acceleration or multi-NPU result.
- **Container status on the tested HPC target:** blocked for an ordinary user
  because the CPU target had no user-accessible runtime and the tested NPU
  target exposed a Docker client without daemon/socket permission.

The Dockerized environment is therefore the reproducible development
specification, while the restricted HPC target uses an equivalent native
ARM64 runtime.
