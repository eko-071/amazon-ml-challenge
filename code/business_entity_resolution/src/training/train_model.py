"""Train the matcher on both target sources (items 1.2 + 1.3-calibrate).

Loads ``train_candidates_S*.tsv`` (both S2 and S3 — never a single
source/country slice), routes each candidate row to its real source file
by ``S2-``/``S3-`` prefix, extracts the rapidfuzz feature set, holds out
an entity-level validation split (1.3-harness), fits the model, then
calibrates it and sweeps the decision threshold against macro-F_0.5
(1.3-calibrate). Saves ``xgb_model_v3.json`` + ``calibrator_v3.pkl`` +
``threshold_v3.json``.

NOTE: features come from ``common.features`` — shared with
``inference/predict.py`` (2.6). Extend the feature set there, never here.

Run from the project root:
    python code/business_entity_resolution/src/training/train_model.py
    [--cal-method isotonic|sigmoid]
"""

import argparse
import glob
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV

from common.features import FEATURES, EMBED_FEATURES, compute_features, add_embedding_features
from common.labels import attach_is_match
from common.metrics import entity_split, macro_f05_sweep

CAND_PATTERN = "final_results/train_candidates_S*.tsv"
TRUTH_FILE = "dataset/train/train_ground_truth.tsv"
MODEL_OUT = "xgb_model_v3.json"
CALIBRATOR_OUT = "calibrator_v3.pkl"
THRESHOLD_OUT = "threshold_v3.json"


def load_candidates(pattern=CAND_PATTERN):
    files = glob.glob(pattern)
    assert len(files) >= 2, f"Expected candidate files for both target sources, found: {files}"
    cands = pd.concat([pd.read_csv(f, sep="\t", dtype=str).fillna("") for f in files], ignore_index=True)
    cands["candidate_entity_id"] = cands["candidate_entity_ids"].str.split(",")
    cands = cands.explode("candidate_entity_id")
    return cands[cands["candidate_entity_id"] != ""].reset_index(drop=True)


def add_labels(cands, truth_df):
    return attach_is_match(cands, truth_df)


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


def calibrate_and_select_threshold(model, val_df, method="isotonic", seed=43, features=None):
    """Calibrate an already-fit model and sweep the F_0.5 threshold.

    Splits validation entities into calibration / threshold-selection
    halves so both stages aren't tuned on the same rows. Uses
    ``cv='prefit'`` (or ``FrozenEstimator`` on sklearn >= 1.6, where
    prefit is deprecated) so calibration never refits the base model.
    Returns ``(calibrator, best_threshold, best_score)``.
    """
    features = list(features) if features is not None else list(FEATURES)
    val_ids = list(val_df["source1_entity_id"].unique())
    cal_ids = entity_split(val_ids, val_frac=0.5, seed=seed)
    cal_mask = val_df["source1_entity_id"].isin(cal_ids)
    assert cal_mask.any() and (~cal_mask).any(), "Validation split too small to halve — need >= 2 val entities."

    try:
        from sklearn.frozen import FrozenEstimator
        base = FrozenEstimator(model)
        calibrator = CalibratedClassifierCV(base, method=method)
    except ImportError:  # sklearn < 1.6
        calibrator = CalibratedClassifierCV(model, method=method, cv="prefit")
    calibrator.fit(val_df.loc[cal_mask, features], val_df.loc[cal_mask, "is_match"])

    thresh_df = val_df.loc[~cal_mask].copy()
    thresh_df["prob"] = calibrator.predict_proba(thresh_df[features])[:, 1]
    results = macro_f05_sweep(thresh_df, np.arange(0.1, 0.95, 0.025))
    best_thresh = max(results, key=results.get)
    return calibrator, float(best_thresh), float(results[best_thresh])


def main(train_dir="dataset/train", cand_pattern=CAND_PATTERN, model_out=MODEL_OUT,
         calibrator_out=CALIBRATOR_OUT, threshold_out=THRESHOLD_OUT, cal_method="isotonic",
         embed_sidecar=None):
    print("Training multi-source XGBoost model (S2+S3)...")
    truth_df = pd.read_csv(TRUTH_FILE if train_dir == "dataset/train" else f"{train_dir}/train_ground_truth.tsv", sep="\t", dtype=str)

    cands = load_candidates(cand_pattern)
    cands = add_labels(cands, truth_df)

    s1 = pd.read_csv(f"{train_dir}/train_source1.tsv", sep="\t", dtype=str).fillna("")
    s2 = pd.read_csv(f"{train_dir}/train_source2.tsv", sep="\t", dtype=str).fillna("")
    s3 = pd.read_csv(f"{train_dir}/train_source3.tsv", sep="\t", dtype=str).fillna("")
    cands = attach_text(cands, s1, s2, s3)
    cands = compute_features(cands)
    features = list(FEATURES)
    if embed_sidecar is not None:
        sidecar = pd.read_csv(embed_sidecar, sep="\t", dtype=str)
        sidecar["name_embedding_cosine"] = sidecar["name_embedding_cosine"].astype(float)
        cands = add_embedding_features(cands, sidecar)
        features = features + EMBED_FEATURES
    print(f"feature set ({len(features)}): {features}")

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
    model.fit(cands.loc[train_mask, features], cands.loc[train_mask, "is_match"])

    val_df = cands.loc[~train_mask].copy()
    val_df["prob"] = model.predict_proba(val_df[features])[:, 1]
    for t, s in sorted(macro_f05_sweep(val_df, [0.3, 0.5, 0.7, 0.9]).items()):
        print(f"val macro F_0.5 @ {t:.2f}: {s:.4f} (raw scores)")

    calibrator, best_thresh, best_score = calibrate_and_select_threshold(
        model, val_df, method=cal_method, features=features
    )
    print(f"Calibrated ({cal_method}): best threshold {best_thresh:.3f}, "
          f"val macro F_0.5 {best_score:.4f}")

    model.save_model(model_out)
    with open(calibrator_out, "wb") as f:
        pickle.dump(calibrator, f)
    with open(threshold_out, "w") as f:
        json.dump({"threshold": best_thresh, "val_macro_f05": best_score,
                   "method": cal_method, "features": features}, f)
    print(f"Saved {model_out}, {calibrator_out}, {threshold_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default="dataset/train")
    parser.add_argument("--cand-pattern", default=CAND_PATTERN)
    parser.add_argument("--model-out", default=MODEL_OUT)
    parser.add_argument("--calibrator-out", default=CALIBRATOR_OUT)
    parser.add_argument("--threshold-out", default=THRESHOLD_OUT)
    parser.add_argument("--cal-method", default="isotonic", choices=["isotonic", "sigmoid"])
    parser.add_argument("--embed-sidecar", default=None,
                        help="Pair-similarity TSV from blocking/embedding_index.py; "
                             "adds name_embedding_cosine to the feature set.")
    args = parser.parse_args()
    main(train_dir=args.train_dir, cand_pattern=args.cand_pattern,
         model_out=args.model_out, calibrator_out=args.calibrator_out,
         threshold_out=args.threshold_out, cal_method=args.cal_method,
         embed_sidecar=args.embed_sidecar)
