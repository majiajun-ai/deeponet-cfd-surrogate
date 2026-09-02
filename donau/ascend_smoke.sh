#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$PROJECT_ROOT/run_ascend.sh" "$@"

# Request one NPU on an appropriate AI worker. Multi-NPU is not justified for
# this 47k-parameter smoke model.
# dsub -wo -q default --label ai -nl <ascend-node> -R cpu=2,mem=4096MB,npu=1 bash donau/ascend_smoke.sh
