# business_entity_resolution

Run from the project root (`/home/fahad_ali/Code/machine_learning/amazon_ml_challenge`).

## Merge submission (1.1)

```bash
python code/business_entity_resolution/src/submission/merge_submission.py
```

## Validation harness (1.3-harness)

```python
import sys
sys.path.insert(0, 'code/business_entity_resolution/src')
from common.metrics import entity_split, macro_f05_sweep

val_ids = entity_split(s1_ids)  # entity-level, 18% holdout, seed 42
scores = macro_f05_sweep(val_df, thresholds=[0.5, 0.7, 0.9])
```

## Multi-source training (1.2)

```bash
# Candidate generation, once per (source, chunk); writes
# final_results/train_candidates_S{2,3}_{chunk}.tsv
python code/business_entity_resolution/src/training/build_train_candidates.py --target_source 2 --chunk 0 --total_chunks 4
python code/business_entity_resolution/src/training/build_train_candidates.py --target_source 3 --chunk 0 --total_chunks 4

# Train (needs train_candidates_S*.tsv for BOTH sources present); saves xgb_model_v3.json
python code/business_entity_resolution/src/training/train_model.py
```

## Calibration (1.3-calibrate)

Runs inside `train_model.py` (no separate script): fits the calibrator on
one val half, sweeps the threshold on the other, saves `calibrator_v3.pkl`
+ `threshold_v3.json` next to the model. Pick the method explicitly:

```bash
python code/business_entity_resolution/src/training/train_model.py --cal-method isotonic  # large val sets
python code/business_entity_resolution/src/training/train_model.py --cal-method sigmoid   # small val sets
```

## Blocking index cache (3.1)

Fit once per (split, source), then all chunked blocking runs load instead
of refitting. The fit phase also builds the numeric-token index (2.4).
Without cache, `generate_candidates.py` falls back to fitting inline
(same result, ~4x the fitting cost).

```bash
python code/business_entity_resolution/src/blocking/fit_target_index.py --split test --target_source 2
python code/business_entity_resolution/src/blocking/fit_target_index.py --split test --target_source 3
# chunks pick it up automatically via --cache-dir (default: cache/)
python code/business_entity_resolution/src/blocking/generate_candidates.py --target_source 2 --chunk 0
```

## Transliteration (2.1)

`common/normalize.py`: `full_normalize()` = `transliterate_to_latin()` +
`normalize_text()`. Deterministic rule-based script conversion covering
all 8 Indic scripts audited in S2/S3 (Devanagari ~9-11%, Kannada/Telugu/
Tamil/Gujarati/Bengali ~1% each, Malayalam ~0.5%, Gurmukhi ~0.25%);
Latin/LatinExt passes through untouched. After adding the dependency:

```bash
uv sync --directory code/business_entity_resolution/src  # picks up indic-transliteration
```

## Numeric-token blocking (2.4)

`blocking/numeric_index.py`: inverted index over 2+ digit address tokens
(`PIN`/street numbers), capped at `max_df=0.05`. Built once per corpus by
`fit_target_index.py` (`{prefix}_numeric.pkl`), loaded by
`generate_candidates.py` (inline single-pass build as fallback), and built
inline by `build_train_candidates.py` so train/test blocking stay
consistent. Language-agnostic — the signal that transfers to unseen
conventions.

## Shared features + inference (2.6-extract)

`common/features.py`: the 8 rapidfuzz similarity features, imported by
both training and inference (no copies). `inference/predict.py` scores
candidates with the calibrated probabilities + swept threshold:

```bash
python code/business_entity_resolution/src/inference/predict.py --target_source 2 --chunk 0
# inputs default to output/multipass_cands_s2_0.tsv, calibrator_v3.pkl, threshold_v3.json;
# output defaults to output/matching_results_s2_0.tsv (override with --cand-file/--out-file)
```
