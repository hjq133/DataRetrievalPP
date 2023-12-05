'''
这个是encode output的, 这个时候记得加上bge的前缀
'''

import os
import json
import pickle
import argparse
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world_size", type=int, required=True)
    parser.add_argument("--dataset_num", type=int, required=True)
    return parser.parse_args()

args = parse_args()

# Function to combine instruction and input
def combine_instruction_input(data):
    '''bge没有prompt前缀'''
    return data['output']

root = args.root
dataset_name = args.dataset_name
output_root = args.output_root
os.makedirs(output_root, exist_ok=True)

model_path = '/mnt/lustre/share_data/huangjunqin/Embedding/HF_Model/bge-base-en-v1.5/'
model = SentenceTransformer(model_path).cuda()

embedding_dict = {}
chunk_size = args.dataset_num // args.world_size + 1
start_idx, end_idx = args.rank * chunk_size, (args.rank + 1) * chunk_size 

with open(os.path.join(root, f'{dataset_name}.jsonl')) as f:
    print(f'process rank {args.rank} start to process {dataset_name} from {start_idx} to {end_idx} ')
    for idx, line in enumerate(tqdm(f.readlines())):
        if start_idx <= idx < end_idx:
            data = json.loads(line.strip())
            text = combine_instruction_input(data)
            emb = model.encode(text, normalize_embeddings=True)
            embedding_dict[idx] = emb

    print(f'process done ..., total items {len(embedding_dict)}')

# Save the dictionary as a pickle file
with open(os.path.join(output_root, f'{dataset_name}_p{args.rank}.pkl'), 'wb') as pkl_file:
    pickle.dump(embedding_dict, pkl_file)
