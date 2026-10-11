# Signer-independent Ghanaian Sign Language category recognition

Code for a controlled study of **signer-independent category recognition** on the healthcare [SignTalk-Gh](https://www.kaggle.com/datasets/responsibleailab/signtalk-ghana) corpus (five signers, 26 thematic categories).

The pipeline extracts MediaPipe Holistic landmarks, shoulder-normalizes them, and trains a compact Transformer over 64-frame windows. The released experiments ask whether standard remedies close the gap to a new signer, and, when they do not, what a short enrollment session actually buys.

## Questions

| ID | Question | Entry point |
| --- | --- | --- |
| RQ1 | Does masked-pose pretraining beat an identical encoder trained from scratch on held-out-signer **sentence** ID? | `scripts/03_pretrain_ssl.py`, `scripts/04_finetune.py`, `scripts/05_evaluate.py` |
| RQ2 | Does error-guided **category** recording beat a matched random category budget? | `scripts/06_simulate_active_learning.py` |
| RQ3 | Does held-out category accuracy rise as more of these five signers are added to training? | `scripts/07_signer_scaling.py` |
| RQ4 | How does accuracy change with a small labeled set from the new signer? | `scripts/09_fewshot_calibration.py` |
| RQ5 | Do landmark-space pseudo-signers replace missing real signers at matched volume? | `scripts/12_synthetic_signer_scaling.py` |
| RQ6 | Do frame replacement or joint masking during fine-tuning help? | `scripts/11_finetune_augment.py` |

RQ1–RQ3 and RQ5–RQ6 are the source-side and augmentation comparisons. The publication revision focuses on **target-signer calibration** (below).

## Calibration revision

This is the comparison used for the revised paper. It reuses the five existing scratch k=4 base checkpoints (`checkpoints/rq4/base_t{1..5}_k4_scratch.pt`). It does not retrain those bases and does not start a new SSL run.

For each of five held-out signers, five calibration draws, and budgets of 1, 5, and 20 clips per available category (75 selections), it scores:

- unadapted base (all 26 outputs, and outputs limited to calibration-present categories)
- nearest-centroid prototypes
- cosine 1-nearest neighbour (cosine 3-NN at 5 and 20 clips only)
- head-only fine-tuning (existing head, encoder frozen; seeds 42, 43, 44)
- partial fine-tuning (head plus the last two encoder layers; same seeds)

```bash
python -u scripts/revision_audit_inputs.py
python -u scripts/revision_historical_replay.py
python -u scripts/revision_run_calibration.py          # resumes finished cells
python -u scripts/revision_summarize.py
python -u scripts/revision_confirm_reporting.py
```

`±` in `results/paper1_access_revision/tables/summary.json` is the sample standard deviation **across the five signer means** (draws averaged inside a signer; adaptation seeds averaged inside a draw). That is not the older 25-run pooled standard deviation.

Reported clip top-1 (mean of five signer means), matched to calibration-present categories where noted:

| Method | 1 clip/category | 5 | 20 |
| --- | ---: | ---: | ---: |
| Base, 26-way | 4.0% | 4.0% | 4.0% |
| Base, calibration categories | 11.8% | 11.8% | 11.8% |
| Prototypes | 34.9% | 50.3% | 54.3% |
| Cosine 1-NN | 34.9% | 54.5% | 64.6% |
| Head-only, 26-way | 4.6% | 7.5% | 32.7% |
| Partial FT, 26-way | 10.7% | 49.5% | 69.3% |

At one clip per category, prototypes and 1-NN agree. At twenty clips, partial fine-tuning is ahead of 1-NN on four of five signers (mean about +4.7 points). Macro-F1 for these same runs is in `results/paper1_access_revision/tables/reporting_confirmation.json`.

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
python scripts/download_mp_models.py
pytest
python scripts/00_smoke_test.py
```

Videos are not in this repository. Download SignTalk-Gh with the steps in `scripts/00_get_data.md`, then:

```bash
python scripts/01_extract_landmarks.py
python scripts/02_build_windows.py
```

Landmark arrays, checkpoints, and full prediction dumps stay local (see `.gitignore`). Summary tables for the calibration revision are included.

## Layout

```
src/                 pose model, SSL, splits, calibration, revision baselines
scripts/             RQ1–RQ6 runners and the revision audit / replay / grid
tests/               pipeline and calibration unit tests
results/paper1_access_revision/tables/   signer-level means, pairs, macro-F1
```

## What this repository does not include

- SignTalk-Gh videos, extracted landmarks, and MediaPipe `.task` files
- Trained checkpoints (including `ssl_encoder.pt` and the RQ4 bases)
- Per-clip prediction dumps and adapted `.pt` files from the 450 fine-tunes
- Draft manuscript notes
