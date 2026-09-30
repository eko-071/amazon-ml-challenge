"""Multilingual embedding similarity as a matcher-side signal (item 2.2, matcher-only slice).

Full ANN retrieval over the 5M-row corpus stays deferred (needs GPU +
FAISS + ~8GB embedding storage). This module covers the cheap, high-value
slice from spec 2.6-step-3: encode unique entity texts once, compute
cosine per post-blocking candidate pair, persist a sidecar file that
training/inference join as the ``name_embedding_cosine`` feature.

The encoder is duck-typed (anything with ``.encode(texts, batch_size,
...)`` returning L2-normalized rows, like
``sentence_transformers.SentenceTransformer`` with
``normalize_embeddings=True``) so the plumbing is testable without the
470MB model. Real-model runs need ``sentence-transformers`` installed
(see pyproject) and happen where there is disk/GPU.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # Apache-2.0, ~118M params


def load_encoder(model_name=DEFAULT_MODEL):
    """Load the real sentence encoder (lazy import — needs installed dep)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def encode_texts(texts, encoder, batch_size=256):
    """Encode a list of texts; return L2-normalized float32 rows."""
    vecs = np.asarray(
        encoder.encode(list(texts), batch_size=batch_size,
                       normalize_embeddings=True, show_progress_bar=False),
        dtype=np.float32,
    )
    return vecs


def build_sidecar(pairs_df, text_lookup, encoder, batch_size=256):
    """Cosine similarity per (source1, candidate) pair.

    ``pairs_df`` has ``source1_entity_id``/``candidate_entity_id``;
    ``text_lookup`` maps entity_id -> text. Each unique entity is
    encoded once regardless of how many pairs mention it.
    Returns ``source1_entity_id, candidate_entity_id,
    name_embedding_cosine``.
    """
    ids = pd.unique(pairs_df[["source1_entity_id", "candidate_entity_id"]].values.ravel())
    missing = [i for i in ids if i not in text_lookup]
    assert not missing, f"{len(missing)} pair IDs without text, e.g. {missing[:5]}"
    embs = encode_texts([text_lookup[i] for i in ids], encoder, batch_size=batch_size)
    mat = dict(zip(ids, embs))
    a = np.stack([mat[i] for i in pairs_df["source1_entity_id"]])
    b = np.stack([mat[i] for i in pairs_df["candidate_entity_id"]])
    out = pairs_df[["source1_entity_id", "candidate_entity_id"]].copy()
    out["name_embedding_cosine"] = (a * b).sum(axis=1)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True,
                        help="Candidate TSV with source1_entity_id,candidate_entity_ids (comma lists).")
    parser.add_argument("--source1-tsv", required=True)
    parser.add_argument("--target-tsv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    s1 = pd.read_csv(args.source1_tsv, sep="\t", dtype=str).fillna("")
    st = pd.read_csv(args.target_tsv, sep="\t", dtype=str).fillna("")
    pairs = pd.read_csv(args.pairs, sep="\t", dtype=str).fillna("")
    pairs["candidate_entity_id"] = pairs["candidate_entity_ids"].str.split(",")
    pairs = pairs.explode("candidate_entity_id")
    pairs = pairs[pairs["candidate_entity_id"] != ""].reset_index(drop=True)

    lookup = dict(zip(s1["entity_id"], s1["business_name"])) | \
        dict(zip(st["entity_id"], st["business_name"]))
    sidecar = build_sidecar(pairs[["source1_entity_id", "candidate_entity_id"]],
                            lookup, load_encoder(args.model), batch_size=args.batch_size)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    sidecar.to_csv(args.out, sep="\t", index=False)
    print(f"Wrote {len(sidecar)} pair similarities -> {args.out}")


if __name__ == "__main__":
    main()
