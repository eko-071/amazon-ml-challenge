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
