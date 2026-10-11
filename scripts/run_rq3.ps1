# RQ3 Step 0: leave-one-signer-out category scaling (no new recordings).
#
#   cd C:\Users\jeffe\OneDrive\Desktop\GSL
#   python -u scripts/07_signer_scaling.py --dry-run
#   python -u scripts/07_signer_scaling.py --quick
#
# Full Step 0 (resume-safe): 5 LOSO folds, 2 subset draws, k=1..4, scratch on
# all folds, SSL arm only on signer 5 (the SSL holdout). Val is 20% of each
# training signer's clips (not a whole extra signer), so k=1 has a real val set.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

Write-Host "=== RQ3 Step 0 dry-run (job grid) ==="
python -u scripts/07_signer_scaling.py --dry-run --draws 2 --epochs 10

Write-Host "=== RQ3 Step 0 LOSO category / clip scaling ==="
python -u scripts/07_signer_scaling.py --draws 2 --epochs 10

Write-Host "=== Figures ==="
python -u scripts/08_make_figures.py --results-only

Write-Host "Done. See results\rq3_signer_scaling.json and results\figures\fig7_signer_diversity_scaling.png"
