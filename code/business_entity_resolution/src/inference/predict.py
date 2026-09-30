"""Test-time inference (items 2.6 + 1.3-calibrate).

Moved from ``run_inference.py``. Features come from ``common.features``
(shared with training — the 2.6 fix), and decisions use the calibrated
probabilities + swept threshold (``calibrator_v3.pkl`` /
``threshold_v3.json`` — the calibrator pickle already wraps the model),
never raw ``model.predict()`` at 0.5.

Run from the project root, once per (target source, chunk):
    python code/business_entity_resolution/src/inference/predict.py \\
        --target_source 2 --chunk 0
"""

import argparse
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from common.features import FEATURES, EMBED_FEATURES, compute_features, add_embedding_features


def predict_chunk(s1_df, st_df, cands_df, calibrator, threshold, features=None, embed_sidecar_df=None):
    """Score one chunk's candidates; return per-S1 match frame."""
    cands = cands_df.copy()
    cands["candidate_entity_id"] = cands["candidate_entity_ids"].str.split(",")
    cands = cands.explode("candidate_entity_id")
    cands = cands[cands["candidate_entity_id"] != ""]
    if cands.empty:
        return pd.DataFrame(columns=["source1_entity_id", "matched_entity_ids"])

    cands = cands.merge(
        s1_df[["entity_id", "business_name", "business_address"]],
        left_on="source1_entity_id", right_on="entity_id",
    ).drop(columns=["entity_id"]).rename(
        columns={"business_name": "business_name_s1", "business_address": "business_address_s1"}
    )
    cands = cands.merge(
        st_df[["entity_id", "business_name", "business_address"]],
        left_on="candidate_entity_id", right_on="entity_id",
    ).drop(columns=["entity_id"]).rename(
        columns={"business_name": "business_name_st", "business_address": "business_address_st"}
    )
    cands = compute_features(cands)
    features = list(features) if features is not None else list(FEATURES)
    if embed_sidecar_df is not None:
        cands = add_embedding_features(cands, embed_sidecar_df)
        assert set(EMBED_FEATURES) <= set(features), "sidecar given but model feature set lacks embedding columns"

    cands["prob"] = calibrator.predict_proba(cands[features])[:, 1]
    hits = cands[cands["prob"] >= threshold]
    matches = (
        hits.groupby("source1_entity_id")["candidate_entity_id"]
        .apply(lambda x: ",".join(sorted(x.dropna().unique())))
        .reset_index()
        .rename(columns={"candidate_entity_id": "matched_entity_ids"})
    )
    return matches


def main(test_dir="dataset/test", target_source=2, chunk=0,
         cand_file=None, calibrator_path="calibrator_v3.pkl",
         threshold_path="threshold_v3.json", out_file=None,
         embed_sidecar=None):
    print(f"--- INFERENCE | Target: S{target_source} | Chunk: {chunk} ---", flush=True)

    s1 = pd.read_csv(f"{test_dir}/test_source1.tsv", sep="\t", dtype=str).fillna("")
    st = pd.read_csv(f"{test_dir}/test_source{target_source}.tsv", sep="\t", dtype=str).fillna("")
    cand_file = cand_file or f"output/multipass_cands_s{target_source}_{chunk}.tsv"
    cands = pd.read_csv(cand_file, sep="\t", dtype=str).fillna("")

    with open(calibrator_path, "rb") as f:
        calibrator = pickle.load(f)
    with open(threshold_path) as f:
        thresh_meta = json.load(f)
    threshold = thresh_meta["threshold"]
    # Train/inference symmetry gate: the feature set recorded at train
    # time must match this run's sidecar state.
    train_features = thresh_meta.get("features", list(FEATURES))
    want_embed = set(EMBED_FEATURES) <= set(train_features)
    if want_embed != (embed_sidecar is not None):
        raise RuntimeError(
            f"Feature mismatch: model trained with features {train_features} "
            f"but --embed-sidecar {'given' if embed_sidecar is not None else 'missing'} — "
            "train and inference must agree."
        )
    sidecar_df = None
    if embed_sidecar is not None:
        sidecar_df = pd.read_csv(embed_sidecar, sep="\t", dtype=str)
        sidecar_df["name_embedding_cosine"] = sidecar_df["name_embedding_cosine"].astype(float)
    print(f"Loaded calibrator; threshold {threshold:.3f}; features {train_features}", flush=True)

    matches = predict_chunk(s1, st, cands, calibrator, threshold,
                            features=train_features, embed_sidecar_df=sidecar_df)

    out_file = out_file or f"output/matching_results_s{target_source}_{chunk}.tsv"
    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    matches.to_csv(out_file, sep="\t", index=False)
    print(f"Inference complete: {len(matches)} S1 with matches -> {out_file}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_source", type=int, required=True)
    parser.add_argument("--chunk", type=int, required=True)
    parser.add_argument("--test-dir", default="dataset/test")
    parser.add_argument("--cand-file", default=None)
    parser.add_argument("--calibrator", default="calibrator_v3.pkl")
    parser.add_argument("--threshold", default="threshold_v3.json")
    parser.add_argument("--out-file", default=None)
    parser.add_argument("--embed-sidecar", default=None,
                        help="Pair-similarity TSV (same file the model trained with).")
    args = parser.parse_args()
    main(test_dir=args.test_dir, target_source=args.target_source, chunk=args.chunk,
         cand_file=args.cand_file, calibrator_path=args.calibrator,
         threshold_path=args.threshold, out_file=args.out_file,
         embed_sidecar=args.embed_sidecar)
