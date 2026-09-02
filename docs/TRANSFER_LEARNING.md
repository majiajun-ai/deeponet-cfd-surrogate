# Transfer interface and future CFD data

The current public model learns a 2D cylinder velocity field. Its shared
branch/trunk latent representation is organized so a future OpenFOAM/monopile
adapter can change the physical inputs, coordinates, and output head without
claiming that unseen channels were pretrained.

## Future case contract

```text
case_id
U, D, H, Re, turbulence_intensity
coordinates[N, 3]  # metres
velocity[N, 3]     # m/s
pressure[N]        # Pa, with p_ref recorded
time               # seconds
optional Cd, Cl, Cl_RMS, St
```

The intended dimensionless fields are:

```text
x* = x / D, y* = y / D, z* = z / D
t* = t U / D
u* = u / U, v* = v / U, w* = w / U
p* = (p - p_ref) / (0.5 rho U²)
```

## Current transfer smoke

The transfer script widens the current `3 → 2 → [u,v]` interface to a
`7 → 3 → [u,v,w,p]` interface and supports a freeze-trunk strategy. The
smoke status is `READY` for interface construction only. The new `w` and `p`
rows are initialized for later fine-tuning; they are not learned from the
CFDBench checkpoint.

## Proposed experiment

For validated OpenFOAM cases, compare scratch initialization with public
checkpoint initialization at 10, 25, 50, 100, 200, and all available cases.
Keep the geometry/query points, normalization, case-level split, optimizer,
and stopping rule aligned. Evaluate velocity and pressure fields together
with `Cd`, `Cl`, `Cl_RMS`, `St`, and wake metrics.

This is a future research plan, not evidence that a production monopile
surrogate is already complete.
