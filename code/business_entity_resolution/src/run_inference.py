import pandas as pd
import xgboost as xgb
from rapidfuzz import fuzz
import argparse
import re
import os

def extract_nums(text):
    return set(re.findall(r'\d+', text))

parser = argparse.ArgumentParser()
parser.add_argument('--target_source', type=int, required=True)
parser.add_argument('--chunk', type=int, required=True)
args = parser.parse_args()

print(f"--- INFERENCE | Target: S{args.target_source} | Chunk: {args.chunk} ---", flush=True)

s1 = pd.read_csv("dataset/test/test_source1.tsv", sep='\t', dtype=str).fillna("")
st = pd.read_csv(f"dataset/test/test_source{args.target_source}.tsv", sep='\t', dtype=str).fillna("")

cands_file = f"output/multipass_cands_s{args.target_source}_{args.chunk}.tsv"
cands = pd.read_csv(cands_file, sep='\t', dtype=str).fillna("")

cands['candidate_entity_id'] = cands['candidate_entity_ids'].str.split(',')
cands = cands.explode('candidate_entity_id')
cands = cands[cands['candidate_entity_id'] != ""]

cands = cands.merge(s1[['entity_id', 'business_name', 'business_address']], left_on='source1_entity_id', right_on='entity_id')
cands = cands.merge(st[['entity_id', 'business_name', 'business_address']], left_on='candidate_entity_id', right_on='entity_id', suffixes=('_s1', '_st'))

print("Extracting features...", flush=True)
n1 = cands['business_name_s1'].fillna('').str.lower().tolist()
n2 = cands['business_name_st'].fillna('').str.lower().tolist()
a1 = cands['business_address_s1'].fillna('').str.lower().tolist()
a2 = cands['business_address_st'].fillna('').str.lower().tolist()

cands['name_exact'] = [1 if x == y else 0 for x, y in zip(n1, n2)]
cands['name_ratio'] = [fuzz.ratio(x, y) for x, y in zip(n1, n2)]
cands['addr_ratio'] = [fuzz.ratio(x, y) for x, y in zip(a1, a2)]
cands['name_token_set'] = [fuzz.token_set_ratio(x, y) for x, y in zip(n1, n2)]
cands['addr_token_set'] = [fuzz.token_set_ratio(x, y) for x, y in zip(a1, a2)]
cands['name_partial'] = [fuzz.partial_ratio(x, y) for x, y in zip(n1, n2)]
cands['name_len_diff'] = [abs(len(x) - len(y)) for x, y in zip(n1, n2)]

num1 = [extract_nums(x) for x in a1]
num2 = [extract_nums(x) for x in a2]
num_match = []
for nums_s1, nums_s2 in zip(num1, num2):
    if not nums_s1 or not nums_s2:
        num_match.append(-1)
    else:
        num_match.append(1 if nums_s1.intersection(nums_s2) else 0)
cands['addr_num_match'] = num_match

features = [
    'name_exact', 'name_ratio', 'addr_ratio', 'name_token_set', 
    'addr_token_set', 'name_partial', 'name_len_diff', 'addr_num_match'
]

print("Loading XGBoost V2...", flush=True)
model = xgb.XGBClassifier()
model.load_model("xgb_model_v2.json")

print("Predicting...", flush=True)
cands['prediction'] = model.predict(cands[features])

matches = cands[cands['prediction'] == 1].groupby('source1_entity_id')['candidate_entity_id'].apply(lambda x: ','.join(x.dropna().unique())).reset_index()
matches.rename(columns={'candidate_entity_id': 'matched_entity_ids'}, inplace=True)

out_file = f"output/matching_results_s{args.target_source}_{args.chunk}.tsv"
matches.to_csv(out_file, sep='\t', index=False)
print(f"Inference Complete! Saved to {out_file}", flush=True)
