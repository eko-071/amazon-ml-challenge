"""Train the matcher on both target sources (item 1.2).

Loads ``train_candidates_S*.tsv`` (both S2 and S3 — never a single
source/country slice), routes each candidate row to its real source file
by ``S2-``/``S3-`` prefix, extracts the rapidfuzz feature set, holds out
an entity-level validation split (1.3-harness), and saves ``xgb_model_v3.json``.

NOTE: the rapidfuzz feature block below is still duplicated with
``inference/predict.py`` until 2.6 extracts ``common/features.py`` —
do not diverge the two copies in the meantime.

Calibration + threshold sweep happen in 1.3-calibrate, against this
model — this script only reports raw val macro-F_0.5 for reference.

Run from the project root:
    python code/business_entity_resolution/src/training/train_model.py
"""

import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import xgboost as xgb
from rapidfuzz import fuzz

from common.metrics import entity_split, macro_f05_sweep

FEATURES = [
    "name_exact", "name_ratio", "addr_ratio", "name_token_set",
    "addr_token_set", "name_partial", "name_len_diff", "addr_num_match",
]

CAND_PATTERN = "final_results/train_candidates_S*.tsv"
TRUTH_FILE = "dataset/train/train_ground_truth.tsv"
MODEL_OUT = "xgb_model_v3.json"


def load_candidates(pattern=CAND_PATTERN):
    files = glob.glob(pattern)
    assert len(files) >= 2, f"Expected candidate files for both target sources, found: {files}"
    cands = pd.concat([pd.read_csv(f, sep="\t", dtype=str).fillna("") for f in files], ignore_index=True)
    cands["candidate_entity_id"] = cands["candidate_entity_ids"].str.split(",")
    cands = cands.explode("candidate_entity_id")
    return cands[cands["candidate_entity_id"] != ""].reset_index(drop=True)


def add_labels(cands, truth_df):
    links = set(zip(
        truth_df["source1_entity_id"],
        truth_df["matched_entity_ids"].str.split(",").explode(),
    ))
    cands["is_match"] = [
        1 if (src, tgt) in links else 0
        for src, tgt in zip(cands["source1_entity_id"], cands["candidate_entity_id"])
    ]
    return cands


def attach_text(cands, s1_df, s2_df, s3_df):
    """Join name/address text, routing each row by candidate ID prefix."""
    s1_cols = ["entity_id", "business_name", "business_address"]
    cands = cands.merge(s1_df[s1_cols], left_on="source1_entity_id", right_on="entity_id")
    cands = cands.drop(columns=["entity_id"]).rename(
        columns={"business_name": "business_name_s1", "business_address": "business_address_s1"}
    )
    st_cols = ["entity_id", "business_name", "business_address"]
    is_s2 = cands["candidate_entity_id"].str.startswith("S2-")
    out = []
    for frame, mask in ((s2_df, is_s2), (s3_df, ~is_s2)):
        sub = cands[mask].merge(frame[st_cols], left_on="candidate_entity_id", right_on="entity_id")
        out.append(sub.drop(columns=["entity_id"]).rename(
            columns={"business_name": "business_name_st", "business_address": "business_address_st"}
        ))
    merged = pd.concat(out, ignore_index=True)
    merged["source"] = merged["candidate_entity_id"].str[:2]  # S2 / S3
    return merged


def extract_nums(text):
    return set(re.findall(r"\d+", text))


def compute_features(cands):
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


def main(train_dir="dataset/train", cand_pattern=CAND_PATTERN, model_out=MODEL_OUT):
    print("Training multi-source XGBoost model (S2+S3)...")
    truth_df = pd.read_csv(TRUTH_FILE if train_dir == "dataset/train" else f"{train_dir}/train_ground_truth.tsv", sep="\t", dtype=str)

    cands = load_candidates(cand_pattern)
    cands = add_labels(cands, truth_df)

    s1 = pd.read_csv(f"{train_dir}/train_source1.tsv", sep="\t", dtype=str).fillna("")
    s2 = pd.read_csv(f"{train_dir}/train_source2.tsv", sep="\t", dtype=str).fillna("")
    s3 = pd.read_csv(f"{train_dir}/train_source3.tsv", sep="\t", dtype=str).fillna("")
    cands = attach_text(cands, s1, s2, s3)
    cands = compute_features(cands)

    print(cands.groupby(["source", "country"])["is_match"].agg(["count", "mean"]))

    val_ids = entity_split(cands["source1_entity_id"].unique())
    train_mask = ~cands["source1_entity_id"].isin(val_ids)

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.05,
        tree_method="hist",
        scale_pos_weight=0.8,
        random_state=42,
    )
    model.fit(cands.loc[train_mask, FEATURES], cands.loc[train_mask, "is_match"])

    val_df = cands.loc[~train_mask].copy()
    val_df["prob"] = model.predict_proba(val_df[FEATURES])[:, 1]
    for t, s in sorted(macro_f05_sweep(val_df, [0.3, 0.5, 0.7, 0.9]).items()):
        print(f"val macro F_0.5 @ {t:.2f}: {s:.4f} (raw scores — calibrate in 1.3-calibrate)")

    model.save_model(model_out)
    print(f"Model saved to {model_out}")


if __name__ == "__main__":
    main()
