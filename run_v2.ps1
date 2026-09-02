param(
    [string]$Python = "python",
    [switch]$SkipRaw,
    [switch]$SkipTraining
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot
$env:PYTHONHOME = ""
$env:PYTHONPATH = ""

$CaseIds = "0,1,3,4,6,8,9,11,12,14,16,17,19,21,22,24,26,27,29,31,32,34,36,37,39,41,42,44,46,47,48,49"

if (-not $SkipRaw) {
    & $Python "src\audit_public_cases.py" --subset bc --output "metadata\cfdb_case_metadata_audit.json"
    & $Python "src\download_public.py" --subset bc --case-ids $CaseIds --raw-root "raw\CFDBench\cylinder_bc_v2" --manifest-output "raw\CFDBench\source_manifest_v2.json"
}

& $Python "src\prepare_dataset_v2.py" --config "configs\v2_e1.yaml"
& $Python "src\prepare_dataset_v2.py" --config "configs\v2_e4.yaml"
& $Python "src\sanity_check_v2.py" --data "processed_v2\operator_dataset_v2.h5" --output "reports_v2\sanity_report_v2_24x16.json"
& $Python "src\sanity_check_v2.py" --data "processed_v2\operator_dataset_v2_48x32.h5" --output "reports_v2\sanity_report_v2_48x32.json"

if (-not $SkipTraining) {
    & $Python "src\train_v2.py" --config "configs\v2_e1.yaml"
    & $Python "src\train_v2.py" --config "configs\v2_e2.yaml"
    & $Python "src\train_v2.py" --config "configs\v2_e3.yaml"
    & $Python "src\train_v2.py" --config "configs\v2_e4.yaml"
}

& $Python "src\evaluate_v2.py" --data "processed_v2\operator_dataset_v2.h5" --checkpoint "checkpoints_v2\v2_e1_baseline_more_cases\best.pt" --experiment-id "v2_e1_baseline_more_cases"
& $Python "src\evaluate_v2.py" --data "processed_v2\operator_dataset_v2.h5" --checkpoint "checkpoints_v2\v2_e2_larger_mlp\best.pt" --experiment-id "v2_e2_larger_mlp"
& $Python "src\evaluate_v2.py" --data "processed_v2\operator_dataset_v2.h5" --checkpoint "checkpoints_v2\v2_e3_fourier_coordinates\best.pt" --experiment-id "v2_e3_fourier_coordinates"
& $Python "src\evaluate_v2.py" --data "processed_v2\operator_dataset_v2_48x32.h5" --checkpoint "checkpoints_v2\v2_e4_fourier_48x32\best.pt" --experiment-id "v2_e4_fourier_48x32"
& $Python "src\aggregate_v2.py" --reports-root "reports_v2" --output "experiment_summary.csv"
& $Python "src\fine_tune_v2.py" --checkpoint "checkpoints_v2\v2_e4_fourier_48x32\best.pt" --strategy freeze_trunk --branch-dim 7 --trunk-dim 3 --output-channels 4 --dry-run
& $Python "src\visualize_v2.py" --data "processed_v2\operator_dataset_v2_48x32.h5" --checkpoint "checkpoints_v2\v2_e4_fourier_48x32\best.pt" --experiment-id "v2_e4_fourier_48x32"
