#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$PROJECT_ROOT/run_arm64.sh" "$@"

# Example after selecting an idle aarch64 CPU node with dnode:
# dsub -wo -q default -nl <cpu-node> -R cpu=2,mem=4096MB bash donau/arm64_cpu_smoke.sh
