# Step 0: Get the SignTalk-Gh dataset

SignTalk-Gh is ~9,879 healthcare-domain Ghanaian Sign Language videos
(4,031 unique sentences, 5 signers, 26 thematic categories) released by
the Responsible AI Lab. You need **signer ID on every clip** — RQ1 and RQ2
are signer-independent.

## 1. Kaggle API token

1. Open https://www.kaggle.com/settings and create an API token.
2. Save `kaggle.json`:
   - Windows: `C:\Users\<you>\.kaggle\kaggle.json`
   - macOS / Linux: `~/.kaggle/kaggle.json`

```bash
pip install kaggle
```

## 2. Download

From the repository root:

```bash
kaggle datasets download -d responsibleailab/signtalk-ghana -p data/raw_videos --unzip
```

Dataset page: https://www.kaggle.com/datasets/responsibleailab/signtalk-ghana

Paper: Ahene et al., *A large scale multimodal dataset for healthcare domain
Ghanaian sign language translation and retrieval based synthesis*,
Scientific Reports (2026). https://doi.org/10.1038/s41598-026-43478-9

## 3. Confirm metadata

You need a table mapping **video filename → sentence text → signer ID**.

The Kaggle listing describes `Metadata.csv` with sentence id, sentence text,
and category. The paper also lists signer identifiers. After unzipping:

```bash
python scripts/00_inspect_metadata.py
```

That script prints the CSV columns, sample video paths, and whether signer
IDs can be resolved automatically.

If signer IDs are missing from both the CSV and the filenames/folders,
create `data/splits/signer_map.csv`:

```
filename,signer_id
clip_001.mp4,1
clip_002.mp4,2
```

Then point `src/config.py` → `Paths.metadata_path` at the metadata CSV if
auto-discovery does not find it.

## 4. Extra unlabeled video (SSL only)

These corpora are **never** in SignTalk train/val/test. Fine-tuning still
uses SignTalk-Gh sentence labels only. Pretraining (`03_pretrain_ssl.py`)
uses SignTalk train+val **plus** anything with `split=ssl`.

### Fragkiadakis lexicon (recommended)

[Fragkiadakis, Nyst & Nyarko (2021)](https://doi.org/10.5281/zenodo.4533753),
~1,200 isolated GSL signs, one signer (Marco Nyarko), CC BY 4.0. The Zenodo
zip includes OpenPose JSON; **ignore those** and re-extract with MediaPipe
so the feature space matches SignTalk-Gh.

```bash
python scripts/download_fragkiadakis.py
# After SignTalk landmark extraction is idle (do not run in parallel on CPU):
python scripts/01_extract_landmarks.py --source fragkiadakis --cpu --workers 10
```

Writes `data/splits/ssl_index.csv`. `02_build_windows.py` merges it in
automatically. AfriSign is optional: drop GSL-only mp4s under
`data/external/afrisign/` and run `--source afrisign`.

Do **not** dump extra videos into `SignTalk-GH/` or `data/raw_videos/` —
that would mix unlabeled lexicon clips into the labeled index.

## 5. Start small

Do **not** extract all 9,879 videos on the first run. Use `--max-videos`
(see README) until the pipeline finishes end-to-end.
