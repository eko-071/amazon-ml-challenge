import pandas as pd
import xgboost as xgb
from rapidfuzz import fuzz
import re

print("Training Upgraded Precision-Heavy XGBoost Model...")
DATA_DIR = "dataset/train"

# 1. Load Ground Truth
truth_df = pd.read_csv(f"{DATA_DIR}/train_ground_truth.tsv", sep='\t')
truth_df['matched_entity_id'] = truth_df['matched_entity_ids'].str.split(',')
truth_df = truth_df.explode('matched_entity_id')
truth_links = set(zip(truth_df['source1_entity_id'], truth_df['matched_entity_id']))

s1 = pd.read_csv(f"{DATA_DIR}/train_source1.tsv", sep='\t', dtype=str).fillna("")
s2 = pd.read_csv(f"{DATA_DIR}/train_source2.tsv", sep='\t', dtype=str).fillna("")
cands = pd.read_csv("final_results/train_candidates_US_S2.tsv", sep='\t')

cands['candidate_entity_id'] = cands['candidate_entity_ids'].str.split(',')
cands = cands.explode('candidate_entity_id')
cands = cands.merge(s1[['entity_id', 'business_name', 'business_address']], left_on='source1_entity_id', right_on='entity_id')
cands = cands.merge(s2[['entity_id', 'business_name', 'business_address']], left_on='candidate_entity_id', right_on='entity_id', suffixes=('_s1', '_s2'))

print("Extracting advanced features...")
n1 = cands['business_name_s1'].fillna('').str.lower().tolist()
n2 = cands['business_name_s2'].fillna('').str.lower().tolist()
a1 = cands['business_address_s1'].fillna('').str.lower().tolist()
a2 = cands['business_address_s2'].fillna('').str.lower().tolist()

# Core RapidFuzz Features
cands['name_exact'] = [1 if x == y else 0 for x, y in zip(n1, n2)]
cands['name_ratio'] = [fuzz.ratio(x, y) for x, y in zip(n1, n2)]
cands['addr_ratio'] = [fuzz.ratio(x, y) for x, y in zip(a1, a2)]
cands['name_token_set'] = [fuzz.token_set_ratio(x, y) for x, y in zip(n1, n2)]
cands['addr_token_set'] = [fuzz.token_set_ratio(x, y) for x, y in zip(a1, a2)]
cands['name_partial'] = [fuzz.partial_ratio(x, y) for x, y in zip(n1, n2)]
cands['name_len_diff'] = [abs(len(x) - len(y)) for x, y in zip(n1, n2)]

# The Precision Protector: Numerical Address Extraction
def extract_nums(text):
    return set(re.findall(r'\d+', text))

num1 = [extract_nums(x) for x in a1]
num2 = [extract_nums(x) for x in a2]

# If both addresses have numbers, do they share at least one? (0 = No, 1 = Yes, -1 = No numbers present)
num_match = []
for nums_s1, nums_s2 in zip(num1, num2):
    if not nums_s1 or not nums_s2:
        num_match.append(-1)
    else:
        num_match.append(1 if nums_s1.intersection(nums_s2) else 0)

cands['addr_num_match'] = num_match

cands['is_match'] = [1 if (src, tgt) in truth_links else 0 for src, tgt in zip(cands['source1_entity_id'], cands['candidate_entity_id'])]

features = [
    'name_exact', 'name_ratio', 'addr_ratio', 'name_token_set', 
    'addr_token_set', 'name_partial', 'name_len_diff', 'addr_num_match'
]

# Tuning XGBoost to prioritize Precision (scale_pos_weight helps manage class imbalance)
model = xgb.XGBClassifier(
    n_estimators=300, 
    max_depth=7, 
    learning_rate=0.05, 
    tree_method='hist', 
    scale_pos_weight=0.8,  # Penalizes false positives slightly to protect F_0.5
    random_state=42
)
model.fit(cands[features], cands['is_match'])

model.save_model("xgb_model_v2.json")
print("🧠 V2 Precision Model successfully saved to xgb_model_v2.json!")
