import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp
import argparse
import gc
import re
import os
import sys

def normalize_text(text):
    if not isinstance(text, str): return ""
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
    if start == end: return []
    data = csr_mat.data[start:end]
    indices = csr_mat.indices[start:end]
    if len(data) <= k: return indices.tolist()
    top_k_idx = np.argpartition(data, -k)[-k:]
    return indices[top_k_idx].tolist()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_source', type=int, required=True)
    parser.add_argument('--chunk', type=int, required=True)
    parser.add_argument('--total_chunks', type=int, default=4)
    parser.add_argument('--cache-dir', type=str, default='cache')
    args = parser.parse_args()

    print(f"--- BULLETPROOF MULTI-PASS | Target: S{args.target_source} | Chunk: {args.chunk} ---", flush=True)

    s1_df = pd.read_csv("dataset/test/test_source1.tsv", sep='\t', dtype=str).fillna("")
    st_df = pd.read_csv(f"dataset/test/test_source{args.target_source}.tsv", sep='\t', dtype=str).fillna("")

    s1_text = (s1_df['business_name'] + " " + s1_df['business_address']).apply(normalize_text)
    st_text = (st_df['business_name'] + " " + st_df['business_address']).apply(normalize_text)
    st_ids = st_df['entity_id'].values

    chunk_size = int(np.ceil(len(s1_df) / args.total_chunks))
    start_idx = args.chunk * chunk_size
    end_idx = min(start_idx + chunk_size, len(s1_df))

    s1_chunk_text = s1_text.iloc[start_idx:end_idx]
    s1_chunk_ids = s1_df['entity_id'].iloc[start_idx:end_idx].values

    del s1_df, st_df, s1_text
    gc.collect()

    # 3.1: load the one-time fitted index when present instead of refitting
    # per chunk. Run blocking/fit_target_index.py first; without cache this
    # falls back to fitting inline (same result, ~4x the fitting cost).
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'blocking'))
    try:
        from fit_target_index import load_target_index, cache_prefix
        _prefix = cache_prefix(args.cache_dir, 'test', args.target_source)
        word_vec, st_word_mat, char_vec, st_char_mat = load_target_index(_prefix)
        print("Loaded cached target index.", flush=True)
        s1_word_mat = word_vec.transform(s1_chunk_text)
        s1_char_mat = char_vec.transform(s1_chunk_text)
        del word_vec, char_vec
    except (FileNotFoundError, ImportError):
        print("No cached index — fitting inline (run fit_target_index.py to skip this).", flush=True)
        word_vec = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=2, max_df=0.05, dtype=np.float32)
        st_word_mat = word_vec.fit_transform(st_text)
        s1_word_mat = word_vec.transform(s1_chunk_text)

        char_vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(4, 4), min_df=2, max_df=0.01, dtype=np.float32)
        st_char_mat = char_vec.fit_transform(st_text)
        s1_char_mat = char_vec.transform(s1_chunk_text)

        del word_vec, char_vec
    del st_text, s1_chunk_text
    gc.collect()

    BATCH_SIZE = 1000
    TOP_K_PER_NET = 20
    TOTAL_ROWS = s1_word_mat.shape[0]
    out_file = f"output/multipass_cands_s{args.target_source}_{args.chunk}.tsv"
    os.makedirs("output", exist_ok=True)

    with open(out_file, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for i in range(0, TOTAL_ROWS, BATCH_SIZE):
            end = min(i + BATCH_SIZE, TOTAL_ROWS)
            w_scores = s1_word_mat[i:end].dot(st_word_mat.T)
            c_scores = s1_char_mat[i:end].dot(st_char_mat.T)
            for row_idx in range(end - i):
                w_top = get_top_k(w_scores, row_idx, TOP_K_PER_NET)
                c_top = get_top_k(c_scores, row_idx, TOP_K_PER_NET)
                union_idx = set(w_top).union(set(c_top))
                cands = [st_ids[idx] for idx in union_idx]
                s1_id = s1_chunk_ids[i + row_idx]
                f.write(f"{s1_id}\t{','.join(cands)}\n")
            if (i // BATCH_SIZE) % 5 == 0:
                print(f"Progress: {(i/TOTAL_ROWS)*100:.1f}%", flush=True)

    print(f"Chunk {args.chunk} Complete! Saved to {out_file}", flush=True)
