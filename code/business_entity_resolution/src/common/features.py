"""Shared matcher features (item 2.6).

Single home for the rapidfuzz feature block previously copy-pasted
between training and inference — the duplication behind the
train/inference skew risk. Both sides import from here; do not fork.

When 2.2 lands, the embedding cosine(s) extend ``FEATURES`` here (and
only here), so both sides gain the signal together.
"""

import re

from rapidfuzz import fuzz

FEATURES = [
    "name_exact", "name_ratio", "addr_ratio", "name_token_set",
    "addr_token_set", "name_partial", "name_len_diff", "addr_num_match",
]

# Optional extension from blocking/embedding_index.py sidecars. Joined,
# never computed, here — so this module stays free of model dependencies.
EMBED_FEATURES = ["name_embedding_cosine"]


def extract_nums(text):
    return set(re.findall(r"\d+", text))


def compute_features(cands):
    """Add the 8 similarity features in place; return the frame.

    Expects ``business_name_s1`` / ``business_address_s1`` (Source-1 side)
    and ``business_name_st`` / ``business_address_st`` (candidate side).
    """
    n1 = cands["business_name_s1"].fillna("").str.lower().tolist()
    n2 = cands["business_name_st"].fillna("").str.lower().tolist()
    a1 = cands["business_address_s1"].fillna("").str.lower().tolist()
    a2 = cands["business_address_st"].fillna("").str.lower().tolist()

    cands["name_exact"] = [1 if x == y else 0 for x, y in zip(n1, n2)]
    cands["name_ratio"] = [fuzz.ratio(x, y) for x, y in zip(n1, n2)]
    cands["addr_ratio"] = [fuzz.ratio(x, y) for x, y in zip(a1, a2)]
    cands["name_token_set"] = [fuzz.token_set_ratio(x, y) for x, y in zip(n1, n2)]
    cands["addr_token_set"] = [fuzz.token_set_ratio(x, y) for x, y in zip(a1, a2)]
    cands["name_partial"] = [fuzz.partial_ratio(x, y) for x, y in zip(n1, n2)]
    cands["name_len_diff"] = [abs(len(x) - len(y)) for x, y in zip(n1, n2)]

    num1 = [extract_nums(x) for x in a1]
    num2 = [extract_nums(x) for x in a2]
    num_match = []
    for nums_s1, nums_st in zip(num1, num2):
        if not nums_s1 or not nums_st:
            num_match.append(-1)
        else:
            num_match.append(1 if nums_s1.intersection(nums_st) else 0)
    cands["addr_num_match"] = num_match
    return cands


def add_embedding_features(cands, sidecar_df):
    """Join precomputed pair cosine(s) onto the candidate frame.

    ``sidecar_df`` has ``source1_entity_id``/``candidate_entity_id``/
    ``name_embedding_cosine`` (see ``blocking/embedding_index.py``).
    Raises if any candidate pair lacks a similarity — a stale sidecar
    must fail loudly, never silently dilute into NaNs.
    """
    before = len(cands)
    cands = cands.merge(
        sidecar_df[["source1_entity_id", "candidate_entity_id"] + EMBED_FEATURES],
        on=["source1_entity_id", "candidate_entity_id"],
        how="left",
    )
    assert len(cands) == before, "sidecar join duplicated rows — rebuild it from these candidates"
    missing = cands[EMBED_FEATURES[0]].isna().sum()
    assert missing == 0, f"sidecar covers {before - missing}/{before} pairs — rebuild it from these candidates"
    return cands
