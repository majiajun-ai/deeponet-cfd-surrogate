#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

EPOCHS="${EPOCHS:-1}"
RUN_NAME="${RUN_NAME:-huawei_arm64_smoke}"

python3 --version
python3 -c 'import platform, torch; print("machine=" + platform.machine()); print("torch=" + str(torch.__version__)); print("cuda=" + str(torch.cuda.is_available())); assert platform.machine() == "aarch64"'
python3 -c 'import h5py, numpy, scipy, yaml; print("data_dependencies=OK")'
python3 -m pip check

python3 src/train_v2.py \
  --config configs/migration_arm64_cpu.yaml \
  --data processed_v2/operator_dataset_v2.h5 \
  --epochs "$EPOCHS" \
  --run-name "$RUN_NAME"
