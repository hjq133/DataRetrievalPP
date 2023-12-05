import pickle
import json

origin_json = '/mnt/lustre/huangjunqin/NLPSpace/LLM/REFINE_DT/trick_tools_v2/filter_long_data/filter_long_zh.jsonl'
keep_idx_path = '/mnt/lustre/huangjunqin/NLPSpace/LLM/REFINE_DT/trick_tools_v2/final_output/search_v8/dedup_zh/0.7/MERGE/0.7_idx.pkl'
output_path = '/mnt/lustre/huangjunqin/NLPSpace/LLM/REFINE_DT/trick_tools_v2/final_output/search_v8/dedup_zh/0.7/MERGE/res_07_zh.json'
dataset_file = open(output_path, 'w')

origin_data = []
with open(origin_json, 'r') as f:
    origin_data = [line for line in f.readlines()]

with open(keep_idx_path, 'rb') as pkl_file:
    filter_idx = list(pickle.load(pkl_file))
    
for idx in filter_idx:
    item = json.loads(origin_data[idx].strip())
    dataset_file.write(json.dumps(item) + '\n')
