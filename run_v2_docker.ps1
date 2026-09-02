param(
    [string]$Image = "deeponet-cfd-surrogate:cpu",
    [switch]$SkipRaw,
    [switch]$SkipTraining
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path -LiteralPath (Split-Path -Parent $MyInvocation.MyCommand.Path)).Path
$MountSpec = "type=bind,source=$ProjectRoot,target=/workspace/deeponet-cfd-surrogate"
$DockerCommand = (Get-Command docker.exe -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Source)
if (-not $DockerCommand) {
    $DockerFallback = Join-Path $env:LOCALAPPDATA "Programs\DockerDesktop\resources\bin\docker.exe"
    if (Test-Path -LiteralPath $DockerFallback) {
        $DockerCommand = $DockerFallback
    } else {
        throw "Docker CLI was not found on PATH or at $DockerFallback."
    }
}

function Invoke-ContainerPython {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$PythonArguments
    )

    $DockerArguments = @(
        "run", "--rm",
        "--mount", $MountSpec,
        $Image,
        "python"
    ) + $PythonArguments

    & $DockerCommand @DockerArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Container command failed with exit code $LASTEXITCODE."
    }
}

Set-Location -LiteralPath $ProjectRoot

$CaseIds = "0,1,3,4,6,8,9,11,12,14,16,17,19,21,22,24,26,27,29,31,32,34,36,37,39,41,42,44,46,47,48,49"

if (-not $SkipRaw) {
    Invoke-ContainerPython @(
        "src/audit_public_cases.py",
        "--subset", "bc",
        "--output", "metadata/cfdb_case_metadata_audit.json"
    )
    Invoke-ContainerPython @(
        "src/download_public.py",
        "--subset", "bc",
        "--case-ids", $CaseIds,
        "--raw-root", "raw/CFDBench/cylinder_bc_v2",
        "--manifest-output", "raw/CFDBench/source_manifest_v2.json"
    )
}

Invoke-ContainerPython @("src/prepare_dataset_v2.py", "--config", "configs/v2_e1.yaml")
Invoke-ContainerPython @("src/prepare_dataset_v2.py", "--config", "configs/v2_e4.yaml")
Invoke-ContainerPython @(
    "src/sanity_check_v2.py",
    "--data", "processed_v2/operator_dataset_v2.h5",
    "--output", "reports_v2/sanity_report_v2_24x16.json"
)
Invoke-ContainerPython @(
    "src/sanity_check_v2.py",
    "--data", "processed_v2/operator_dataset_v2_48x32.h5",
    "--output", "reports_v2/sanity_report_v2_48x32.json"
)

if (-not $SkipTraining) {
    foreach ($Config in @("v2_e1.yaml", "v2_e2.yaml", "v2_e3.yaml", "v2_e4.yaml")) {
        Invoke-ContainerPython @("src/train_v2.py", "--config", "configs/$Config")
    }
}

Invoke-ContainerPython @(
    "src/evaluate_v2.py",
    "--data", "processed_v2/operator_dataset_v2.h5",
    "--checkpoint", "checkpoints_v2/v2_e1_baseline_more_cases/best.pt",
    "--experiment-id", "v2_e1_baseline_more_cases"
)
Invoke-ContainerPython @(
    "src/evaluate_v2.py",
    "--data", "processed_v2/operator_dataset_v2.h5",
    "--checkpoint", "checkpoints_v2/v2_e2_larger_mlp/best.pt",
    "--experiment-id", "v2_e2_larger_mlp"
)
Invoke-ContainerPython @(
    "src/evaluate_v2.py",
    "--data", "processed_v2/operator_dataset_v2.h5",
    "--checkpoint", "checkpoints_v2/v2_e3_fourier_coordinates/best.pt",
    "--experiment-id", "v2_e3_fourier_coordinates"
)
Invoke-ContainerPython @(
    "src/evaluate_v2.py",
    "--data", "processed_v2/operator_dataset_v2_48x32.h5",
    "--checkpoint", "checkpoints_v2/v2_e4_fourier_48x32/best.pt",
    "--experiment-id", "v2_e4_fourier_48x32"
)
Invoke-ContainerPython @("src/aggregate_v2.py", "--reports-root", "reports_v2", "--output", "experiment_summary.csv")
Invoke-ContainerPython @(
    "src/fine_tune_v2.py",
    "--checkpoint", "checkpoints_v2/v2_e4_fourier_48x32/best.pt",
    "--strategy", "freeze_trunk",
    "--branch-dim", "7",
    "--trunk-dim", "3",
    "--output-channels", "4",
    "--dry-run"
)
Invoke-ContainerPython @(
    "src/visualize_v2.py",
    "--data", "processed_v2/operator_dataset_v2_48x32.h5",
    "--checkpoint", "checkpoints_v2/v2_e4_fourier_48x32/best.pt",
    "--experiment-id", "v2_e4_fourier_48x32"
)
