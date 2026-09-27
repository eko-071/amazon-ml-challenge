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
