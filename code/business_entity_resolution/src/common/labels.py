"""Ground-truth labeling helpers.

Single home for joining candidate pairs against ground truth, so the
correct pattern is structural rather than something each call site has
to independently remember to get right.
"""

import pandas as pd


def attach_is_match(
    cands_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    s1_col: str = "source1_entity_id",
    cand_col: str = "candidate_entity_id",
    matched_col: str = "matched_entity_ids",
) -> pd.DataFrame:
    """Label each row of ``cands_df`` with ``is_match`` (0/1).

    ``truth_df`` has columns ``[s1_col, matched_col]`` where
    ``matched_col`` is a comma-separated string of matched entity ids
    (possibly empty/NaN for singletons).

    Uses explode — not a positional zip of two independently-lengthed
    sequences — so each exploded matched-id stays attached to the
    source1 id it came from. Zipping N source ids against M flattened
    match ids (N != M whenever per-entity match counts differ, i.e. the
    common case) silently misaligns pairs; this function makes that
    bug class unexpressible at the call site.
    """
    truth_long = (
        truth_df.assign(_matched=truth_df[matched_col].fillna("").str.split(","))
        .explode("_matched")
    )
    truth_long = truth_long[truth_long["_matched"].str.strip() != ""]
    true_pairs = set(zip(truth_long[s1_col], truth_long["_matched"]))

    out = cands_df.copy()
    out["is_match"] = [
        1 if (s1, cand) in true_pairs else 0
        for s1, cand in zip(out[s1_col], out[cand_col])
    ]
    return out
