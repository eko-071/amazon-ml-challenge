"""One-time fit+cache phase for blocking indices (item 3.1).

Fits the TF-IDF word/char vectorizers on a target-source corpus once and
persists the fitted vectorizers plus the transformed target matrices, so
chunked query scripts load instead of refitting per chunk (previously
16 redundant fits per full run).

Cache is keyed by ``{split}/{source}`` — train and test corpora must
never share artifacts (different vocabularies, and reuse would leak).

NOTE: normalizes with ``common.normalize.full_normalize`` — the same
function the query phase uses. Caches fitted before 2.1 (Latin-only
``normalize_text``) are stale; refit after this change.

Run from the project root, once per (split, target source):
    python code/business_entity_resolution/src/blocking/fit_target_index.py \\
        --split test --target_source 2
"""

import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from common.normalize import full_normalize
from blocking.numeric_index import build_numeric_index, save_numeric_index


def cache_prefix(cache_dir, split, target_source):
    return os.path.join(cache_dir, split, f"s{target_source}")


def fit_target_index(st_df, prefix):
    """Fit word+char TF-IDF on the target corpus; persist vectorizers + matrices."""
    st_text = (st_df["business_name"].fillna("") + " " + st_df["business_address"].fillna("")).apply(full_normalize)

    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_df=0.05, dtype=np.float32)
    st_word_mat = word_vec.fit_transform(st_text)

    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 4), min_df=2, max_df=0.01, dtype=np.float32)
    st_char_mat = char_vec.fit_transform(st_text)

    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    joblib.dump(word_vec, prefix + "_word_vec.joblib")
    sp.save_npz(prefix + "_word_mat.npz", st_word_mat)
    joblib.dump(char_vec, prefix + "_char_vec.joblib")
    sp.save_npz(prefix + "_char_mat.npz", st_char_mat)
    num_index = build_numeric_index(st_df)
    save_numeric_index(num_index, prefix + "_numeric.pkl")
    return word_vec, st_word_mat, char_vec, st_char_mat


def load_target_index(prefix):
    """Load a previously fitted index; raises FileNotFoundError if absent."""
    word_vec = joblib.load(prefix + "_word_vec.joblib")
    st_word_mat = sp.load_npz(prefix + "_word_mat.npz")
    char_vec = joblib.load(prefix + "_char_vec.joblib")
    st_char_mat = sp.load_npz(prefix + "_char_mat.npz")
    return word_vec, st_word_mat, char_vec, st_char_mat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=["train", "test"])
    parser.add_argument("--target_source", type=int, required=True, choices=[2, 3])
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--cache-dir", default="cache")
    args = parser.parse_args()

    data_dir = args.data_dir or f"dataset/{args.split}"
    st_df = pd.read_csv(f"{data_dir}/{args.split}_source{args.target_source}.tsv", sep="\t", dtype=str).fillna("")
    prefix = cache_prefix(args.cache_dir, args.split, args.target_source)
    _, w_mat, _, c_mat = fit_target_index(st_df, prefix)
    n_tok = len(pickle.load(open(prefix + "_numeric.pkl", "rb")))
    print(f"Fitted on {len(st_df)} rows; word matrix {w_mat.shape}, char matrix {c_mat.shape}, {n_tok} numeric tokens -> {prefix}_*")


if __name__ == "__main__":
    main()
