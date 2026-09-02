# Dataset, code license, and attribution

## Code

The original code in this repository is released under the MIT License. The
license applies to the repository code and documentation authored for this
project; it does not change the terms of any external dataset or upstream
software.

## CFDBench data

The experiment uses the `cylinder/bc` portion of CFDBench. The dataset card
declares Apache-2.0 for the released dataset. The raw archive and generated
HDF5 files are **not redistributed in this repository**. The download script
records source URLs, selected archive members, CRC checks, and SHA-256 hashes
for a local provenance trail.

Before redistributing any data or derived artifact, consult the current
CFDBench dataset card and its accompanying terms. The selected release
contains interpolated `u.npy` and `v.npy` fields only; it does not provide a
pressure field for this experiment.

Sources:

- [CFDBench dataset card](https://huggingface.co/datasets/chen-yingfa/CFDBench)
- [CFDBench upstream repository](https://github.com/luo-yining/CFDBench)

## Citation

Please cite the upstream benchmark when using the data or reproducing the
experiment:

> Luo, Yining, Yingfa Chen, and Zhang. *CFDBench: A Comprehensive CFD
> Benchmark for Deep Learning.* arXiv:2310.05963, 2023.

The project intentionally does not copy upstream source code into this
repository. Any future secondary dataset (for example a pressure-aware
cylinder or 3D bluff-body source) must be reviewed separately for its data
license, citation, and redistribution terms.
