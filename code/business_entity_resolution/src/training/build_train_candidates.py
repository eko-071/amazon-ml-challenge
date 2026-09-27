"""Generate training candidate files for both target sources (item 1.2).

Same blocking logic as ``bulletproof_multipass.py`` but pointed at
``dataset/train/`` instead of ``dataset/test/``. Writes one file per
target source — ``train_candidates_S2.tsv`` / ``train_candidates_S3.tsv``
— with ``country`` kept as a *column*, never a filename partition
(open-set rule: no stage may assume a fixed country list).

NOTE (Tier 0 ordering wrinkle): this duplicates the TF-IDF blocking logic
from the still-flat ``bulletproof_multipass.py`` because 2.1/3.1 haven't
landed yet. Treat that copy as disposable — once ``fit_target_index.py`` /
``generate_candidates.py`` exist, rewrite this to call them instead of
keeping its own copy.

Run from the project root, once per (target source, chunk):
    python code/business_entity_resolution/src/training/build_train_candidates.py \\
        --target_source 2 --chunk 0 --total_chunks 4
"""

import argparse
import gc
import os
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# Disposable copy of bulletproof_multipass.normalize_text — do not extend
# here; 2.1 extracts the real one into common/normalize.py.
def normalize_text(text):
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = re.sub(r'[^\w\s]', ' ', t)
    t = re.sub(r'\b(corporation|corp)\b', 'corp', t)
    t = re.sub(r'\b(limited|ltd)\b', 'ltd', t)
    t = re.sub(r'\b(private|pvt)\b', 'pvt', t)
    t = re.sub(r'\b(company|co)\b', 'co', t)
    t = re.sub(r'\b(road|rd)\b', 'rd', t)
    t = re.sub(r'\b(street|st)\b', 'st', t)
    return re.sub(r'\s+', ' ', t).strip()


def get_top_k(csr_mat, row_idx, k):
    start = csr_mat.indptr[row_idx]
    end = csr_mat.indptr[row_idx + 1]
    if start == end:
        return []
    data = csr_mat.data[start:end]
    indices = csr_mat.indices[start:end]
    if len(data) <= k:
        return indices.tolist()
    top_k_idx = np.argpartition(data, -k)[-k:]
    return indices[top_k_idx].tolist()


def generate_for_source(s1_df, st_df, top_k=20, batch_size=1000):
    """Return a candidate frame for one target source.

    Expects ``s1_df`` with ``entity_id``/``business_name``/``business_address``/
    ``country`` and ``st_df`` with ``entity_id``/``business_name``/
    ``business_address``. Returns columns ``source1_entity_id``,
    ``candidate_entity_ids`` (comma-joined), ``country`` (from the S1 side).
    """
    s1_text = (s1_df["business_name"].fillna("") + " " + s1_df["business_address"].fillna("")).apply(normalize_text)
    st_text = (st_df["business_name"].fillna("") + " " + st_df["business_address"].fillna("")).apply(normalize_text)
    st_ids = st_df["entity_id"].values

    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_df=0.05, dtype=np.float32)
    st_word_mat = word_vec.fit_transform(st_text)
    s1_word_mat = word_vec.transform(s1_text)

    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 4), min_df=2, max_df=0.01, dtype=np.float32)
    st_char_mat = char_vec.fit_transform(st_text)
    s1_char_mat = char_vec.transform(s1_text)

    rows = []
    total = s1_word_mat.shape[0]
    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)
        w_scores = s1_word_mat[i:end].dot(st_word_mat.T)
        c_scores = s1_char_mat[i:end].dot(st_char_mat.T)
        for row_idx in range(end - i):
            union_idx = set(get_top_k(w_scores, row_idx, top_k)).union(
                set(get_top_k(c_scores, row_idx, top_k))
            )
            rows.append({
                "source1_entity_id": s1_df["entity_id"].iloc[i + row_idx],
                "candidate_entity_ids": ",".join(st_ids[idx] for idx in union_idx),
                "country": s1_df["country"].iloc[i + row_idx],
            })
    return pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_ids", "country"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_source", type=int, required=True, choices=[2, 3])
    parser.add_argument("--chunk", type=int, required=True)
    parser.add_argument("--total_chunks", type=int, default=4)
    parser.add_argument("--train_dir", default="dataset/train")
    parser.add_argument("--out_dir", default="final_results")
    parser.add_argument("--top_k", type=int, default=20)
    args = parser.parse_args()

    print(f"--- BUILD TRAIN CANDIDATES | Target: S{args.target_source} | Chunk: {args.chunk} ---", flush=True)

    s1_df = pd.read_csv(f"{args.train_dir}/train_source1.tsv", sep="\t", dtype=str).fillna("")
    st_df = pd.read_csv(f"{args.train_dir}/train_source{args.target_source}.tsv", sep="\t", dtype=str).fillna("")

    chunk_size = int(np.ceil(len(s1_df) / args.total_chunks))
    start = args.chunk * chunk_size
    end = min(start + chunk_size, len(s1_df))
    s1_chunk = s1_df.iloc[start:end].reset_index(drop=True)
    del s1_df
    gc.collect()

    cands = generate_for_source(s1_chunk, st_df, top_k=args.top_k)
    del st_df
    gc.collect()

    os.makedirs(args.out_dir, exist_ok=True)
    out_file = f"{args.out_dir}/train_candidates_S{args.target_source}_{args.chunk}.tsv"
    cands.to_csv(out_file, sep="\t", index=False)
    print(f"Chunk {args.chunk} complete: {len(cands)} S1 rows -> {out_file}", flush=True)


if __name__ == "__main__":
    main()
