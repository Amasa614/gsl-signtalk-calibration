# Remaining paper runs after the current RQ2 category job finishes.
# Run from the repo root in PowerShell (not in Cursor):
#
#   cd C:\Users\jeffe\OneDrive\Desktop\GSL
#   .\scripts\run_remaining.ps1
#
# Do NOT start this while 06_simulate_active_learning.py is still using the GPU.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

$rq2 = "results\rq2_simulation_category_entropy.json"
if (-not (Test-Path $rq2)) {
    Write-Error @"
RQ2 category job is not finished yet (missing $rq2).
Wait until the running command completes, then re-run this script.
"@
}

Write-Host "=== RQ2 figures ==="
python -u scripts/08_make_figures.py --results-only

Write-Host "=== RQ2 signer-addition protocol (§3.5.1) ==="
python -u scripts/06_simulate_active_learning.py `
    --protocol signer `
    --method entropy `
    --seed-signers 1,2 `
    --test-signers 5 `
    --epochs 15 `
    --pretrained checkpoints/ssl_encoder.pt

Write-Host "=== RQ3 signer-diversity scaling ==="
python -u scripts/07_signer_scaling.py `
    --train-signers 1,2,3 `
    --val-signers 4 `
    --test-signers 5 `
    --epochs 15 `
    --pretrained checkpoints/ssl_encoder.pt

Write-Host "=== Refresh figures (RQ2 + RQ3) ==="
python -u scripts/08_make_figures.py --results-only

Write-Host "Done. Check results\figures\ and results\rq2_simulation_*.json / results\rq3_signer_scaling.json"
