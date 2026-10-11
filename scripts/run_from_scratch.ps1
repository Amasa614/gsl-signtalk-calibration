# Full RQ1–RQ3 retrain after Fragkiadakis OpenPose (GSL_openpose_data) is ingested.
# SignTalk MediaPipe landmarks are reused. SSL + fine-tune + RQ2/RQ3 start over.
#
# Run from the repo root in PowerShell. Stop any other training job first
# (the GPU can only hold one of 03/04/06/07 at a time):
#
#   cd C:\Users\jeffe\OneDrive\Desktop\GSL
#   .\scripts\run_from_scratch.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

Write-Host "=== Fragkiadakis OpenPose -> landmark npz (SSL only) ==="
python -u scripts/01_extract_landmarks.py --source fragkiadakis

Write-Host "=== Rebuild windows (SignTalk SI split + SSL lexicon) ==="
python -u scripts/02_build_windows.py --test-signers 5 --val-signers 4

$archive = "results\archive_signtalk_ssl"
if (-not (Test-Path $archive)) {
    New-Item -ItemType Directory -Path $archive | Out-Null
    foreach ($name in @(
        "ssl_history.json",
        "finetune_pretrained_history.json",
        "finetune_scratch_history.json",
        "rq1_comparison.json",
        "rq2_simulation_category_entropy.json",
        "rq2_simulation_signer_entropy.json",
        "rq3_signer_scaling.json"
    )) {
        $src = "results\$name"
        if (Test-Path $src) {
            Copy-Item $src "$archive\$name"
        }
    }
    Write-Host "archived previous JSON under $archive"
}

Write-Host "=== SSL pretrain (SignTalk train/val + Fragkiadakis; test signer held out) ==="
python -u scripts/03_pretrain_ssl.py

Write-Host "=== RQ1 fine-tune pretrained ==="
python -u scripts/04_finetune.py --pretrained checkpoints/ssl_encoder.pt

Write-Host "=== RQ1 fine-tune scratch ==="
python -u scripts/04_finetune.py --pretrained none

Write-Host "=== RQ1 evaluate ==="
python -u scripts/05_evaluate.py
python -u scripts/08_make_figures.py --results-only

Write-Host "=== RQ2 category protocol ==="
python -u scripts/06_simulate_active_learning.py `
    --protocol category `
    --method entropy `
    --seed-signers 1,2 `
    --test-signers 5 `
    --budget 4 `
    --epochs 15 `
    --pretrained checkpoints/ssl_encoder.pt

Write-Host "=== RQ2 signer protocol ==="
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

Write-Host "=== Refresh figures ==="
python -u scripts/08_make_figures.py --results-only

Write-Host "Done. Compare new JSON in results\ vs results\archive_signtalk_ssl\"
