# Containerization and multi-architecture workflow

## Image responsibilities

The default `Dockerfile` defines a CPU-oriented Python 3.11 runtime with
CPU-only PyTorch, scientific Python dependencies, and the project source.
Raw data, HDF5 datasets, checkpoints, logs, and generated figures remain on
the host. This keeps the image reproducible without baking multi-gigabyte
scientific artifacts into it.

`compose.yaml` defines a reproducible training-task configuration: image build,
working directory, source bind mount, and `PYTHONPATH`. In short:

```text
Dockerfile = how the image is built
Compose    = how the container is run
```

The separate `Dockerfile.multistage` demonstrates a builder stage, a runtime
stage, and `COPY --from` to separate dependency installation from runtime
assembly. It is an engineering reference; no unsupported image-size claim is
made.

## Data boundary

The `.dockerignore` excludes raw CFD data, processed HDF5 files, checkpoints,
logs, reports, figures, and common scientific archives. The source experiment
kept roughly 2.3 GB of raw CFD/runtime artifacts outside the image; BuildKit
reported a 2.70 KB context for the verified multi-architecture source build.
Context size is checkout-dependent and should be measured again after local
changes.

Example bind mount for a prepared dataset:

```powershell
docker run --rm `
  --mount "type=bind,source=$((Get-Location).Path)\processed_v2,target=/workspace/deeponet-cfd-surrogate/processed_v2,readonly" `
  deeponet-cfd-surrogate:cpu `
  python src/train_v2.py --config configs/v2_e1.yaml --epochs 1 --run-name docker_smoke
```

## Multi-platform publication

Buildx can publish one tag containing both Linux variants:

```powershell
docker buildx build `
  --platform linux/amd64,linux/arm64 `
  --push `
  --tag <registry-user>/deeponet-cfd-surrogate:latest .

docker buildx imagetools inspect <registry-user>/deeponet-cfd-surrogate:latest
```

The registry manifest selects `linux/amd64` on an amd64 host and `linux/arm64`
on an ARM64 host. An x86 host can use ARM64 emulation for compatibility
checks, but native ARM64 builds are normally faster.

## Huawei runtime boundary

The tested HPC target did not expose a Docker daemon to an ordinary user. The
working replacement was native ARM64 Python with platform-provided PyTorch,
`torch_npu`/CANN where applicable, Donau scheduling, and external data/output
paths. This is portability validation, not a claim of container deployment on
the target cluster.
