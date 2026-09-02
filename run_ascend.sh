#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

# The worker normally exports CANN through its job environment. Source a
# versioned fallback only when the scheduler did not do so.
if [[ -n "${ASCEND_TOOLKIT_HOME:-}" && -f "$ASCEND_TOOLKIT_HOME/set_env.sh" ]]; then
  # shellcheck disable=SC1090
  source "$ASCEND_TOOLKIT_HOME/set_env.sh"
elif [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
elif [[ -f /usr/local/Ascend/ascend-toolkit/8.0.0/set_env.sh ]]; then
  source /usr/local/Ascend/ascend-toolkit/8.0.0/set_env.sh
fi

EPOCHS="${EPOCHS:-1}"
RUN_NAME="${RUN_NAME:-huawei_ascend_smoke}"

python3 --version
python3 -c 'import platform, torch, torch_npu; print("machine=" + platform.machine()); print("torch=" + str(torch.__version__)); print("torch_npu=" + str(torch_npu.__version__)); print("npu_available=" + str(torch.npu.is_available())); print("npu_count=" + str(torch.npu.device_count())); assert platform.machine() == "aarch64"; assert torch.npu.is_available()'
python3 -c 'import torch, torch_npu; x = torch.ones((2, 3)).npu(); y = x * 2; torch.npu.synchronize(); print("tensor=" + str(y.cpu()))'
python3 -c 'import h5py, numpy, scipy, yaml; print("data_dependencies=OK")'
if pip_check_output="$(python3 -m pip check 2>&1)"; then
  echo "pip_check=PASS"
else
  expected_platform_issue=$'op-compile-tool 0.1.0 requires getopt, which is not installed.\nop-compile-tool 0.1.0 requires inspect, which is not installed.\nop-compile-tool 0.1.0 requires multiprocessing, which is not installed.'
  if [[ "$pip_check_output" == "$expected_platform_issue" ]]; then
    printf '%s\n' "$pip_check_output"
    echo "pip_check=KNOWN_PLATFORM_METADATA_ONLY"
  else
    printf '%s\n' "$pip_check_output"
    echo "pip_check=FAIL"
    exit 1
  fi
fi

python3 src/train_v2.py \
  --config configs/migration_ascend_npu.yaml \
  --data processed_v2/operator_dataset_v2.h5 \
  --epochs "$EPOCHS" \
  --run-name "$RUN_NAME"
