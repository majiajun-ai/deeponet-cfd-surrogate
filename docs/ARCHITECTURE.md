# Architecture

## Current operator

The project learns a parameter/time-conditioned velocity field with a shared
branch/trunk DeepONet:

```text
branch([Re, U_inlet, time_norm]) ─┐
                                  ├─ latent contraction ─→ [u, v]
trunk([x/D, y/D])                ─┘
```

The selected best configuration uses a larger MLP with four-frequency Fourier
coordinate features:

```text
branch dimension      3
trunk dimension       2
output channels       2 (u, v)
width                 160
depth                 4
latent dimension      80
activation            GELU
Fourier frequencies   4
query grid            48 × 32
trainable parameters  184,162
```

The baseline smoke configuration is a 47,330-parameter model with width 96,
depth 3, latent dimension 48, and no Fourier features. Parameter counts are
computed from the current model configuration and trainable tensors; older
prototype counts are not mixed into the current result table.

## Scaling and normalization

The physical fields are first nondimensionalized:

```text
x* = (x - xc) / D
y* = (y - yc) / D
t* = t U_inlet / D
u* = u / U_inlet
v* = v / U_inlet
```

The network receives `[x*, y*]` as trunk coordinates and a normalized time
coordinate in the branch. Global standardization is fitted on training cases
only and stored as metadata for reproducible inverse scaling.

## Future transfer interface

The transfer smoke test widens the interface to a future 3D/pressure schema:

```text
current: branch 3 → trunk 2 → output 2 [u, v]
future:  branch 7 → trunk 3 → output 4 [u, v, w, p]
```

Shared latent weights can be reused and the output head can be replaced or
partially initialized. The new `w` and `p` channels are not pretrained by the
current cylinder release; they require validated OpenFOAM/monopile data.

## Design boundary

The model is a data-driven baseline. No divergence penalty, pressure loss, or
production CFD solver is claimed in the current public benchmark.
