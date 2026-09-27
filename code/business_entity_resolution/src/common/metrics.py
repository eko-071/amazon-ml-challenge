"""Shared validation harness: entity-level splits + macro-averaged F_0.5.

F_0.5 is computed per Source-1 entity, then averaged — singletons included.
An entity with no true matches scores 1.0 when predicted empty, 0.0 otherwise.
Entities with zero candidate rows at all (complete blocking misses) must be
passed via ``all_entities``/``true_by_entity`` or the score is optimistic.
"""

import numpy as np


def entity_split(s1_ids, val_frac=0.18, seed=42):
    """Split Source-1 entity IDs into a held-out validation set.

    Splits at the entity level, not the candidate-row level, so no
    Source-1 entity appears on both sides (row-level splits leak
    entity-specific patterns into validation).
    """
    ids = np.unique(np.asarray(list(s1_ids), dtype=str))
    n_val = int(round(len(ids) * val_frac))
    rng = np.random.RandomState(seed)
    return set(rng.choice(ids, size=n_val, replace=False).tolist())


def macro_f05_sweep(df_val, thresholds, prob_col="prob",
                    all_entities=None, true_by_entity=None):
    """Score every threshold in one pass; return ``{threshold: macro F_0.5}``.

    ``df_val`` needs ``source1_entity_id``, ``candidate_entity_id``,
    ``is_match`` (0/1) and the probability column. ``true_by_entity`` and
    ``all_entities`` are derived from ``df_val`` unless explicitly given —
    pass both (built from ground truth, not candidates) when some
    validation entities have zero candidate rows.
    """
    if true_by_entity is None:
        true_by_entity = (
            df_val[df_val["is_match"] == 1]
            .groupby("source1_entity_id")["candidate_entity_id"]
            .apply(set)
        )
    if all_entities is None:
        all_entities = df_val["source1_entity_id"].unique()

    results = {}
    for t in thresholds:
        preds = df_val[df_val[prob_col] >= t]
        pred_by_entity = (
            preds.groupby("source1_entity_id")["candidate_entity_id"].apply(set)
        )
        scores = []
        for eid in all_entities:
            pred_set = pred_by_entity.get(eid, set())
            true_set = true_by_entity.get(eid, set())
            if not pred_set and not true_set:
                scores.append(1.0)
            elif not pred_set or not true_set:
                scores.append(0.0)
            else:
                tp = len(pred_set & true_set)
                p = tp / len(pred_set)
                r = tp / len(true_set)
                scores.append(0.0 if (p == 0 and r == 0) else (1.25 * p * r) / (0.25 * p + r))
        results[t] = float(np.mean(scores))
    return results
