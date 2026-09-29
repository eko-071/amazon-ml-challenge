"""Numeric-token blocking net (item 2.4).

Street numbers and postal/PIN codes are near-invariant across language,
script, and formatting — the language-agnostic signal, and the one that
transfers to unseen conventions (e.g. France, test-only). Complements
the lexical TF-IDF nets: a shared distinctive number surfaces pairs
whose surrounding text differs completely.
"""

import os
import pickle
import re

# Consistent with the EDA notebook: 2+ digit sequences only. Single
# digits are too common to carry any blocking signal.
NUMERIC_TOKEN_RE = re.compile(r"\b\d{2,}\b")


def extract_numeric_tokens(text):
    if not isinstance(text, str):
        return []
    return NUMERIC_TOKEN_RE.findall(text)


def build_numeric_index(st_df, max_df=0.05):
    """Build token -> sorted entity-id list from the address field.

    Tokens above ``max_df`` (fraction of records) are dropped — same
    spirit as the TF-IDF caps — so near-useless high-frequency numbers
    can't bloat the candidate pool.
    """
    postings = {}
    for eid, addr in zip(st_df["entity_id"], st_df["business_address"].fillna("")):
        for tok in set(extract_numeric_tokens(addr)):
            postings.setdefault(tok, []).append(eid)
    n = max(len(st_df), 1)
    return {tok: sorted(ids) for tok, ids in postings.items() if len(ids) / n <= max_df}


def lookup_numeric_candidates(tokens, index):
    out = set()
    for tok in tokens:
        ids = index.get(tok)
        if ids:
            out.update(ids)
    return out


def save_numeric_index(index, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(index, f)


def load_numeric_index(path):
    """Load a persisted index; raises FileNotFoundError if absent."""
    with open(path, "rb") as f:
        return pickle.load(f)
