import os
import json
import pickle
import torch
import torch.nn.functional as F
import numpy as np
import argparse

from tqdm import tqdm
from collections import defaultdict
from sentence_transformers import SentenceTransformer

############ hyper parameter
num_file = 18
root = '/mnt/lustre/huangjunqin/NLPSpace/LLM/REFINE_DT/trick_tools_v2/filter_long_data/'
KEY = 'output'
TOPK = 70
dataset_name = 'filter_long_en'
embedding_root = f'/mnt/lustre/huangjunqin/NLPSpace/LLM/REFINE_DT/trick_tools_v2/encode_emb_en/{KEY}'
prefix_instruction = 'Represent this sentence for searching relevant passages:' # only for search docs
query_roots = [
    '/mnt/lustre/huangjunqin/NLPSpace/LLM/challenge-data/dev_for_search/',
    # '/mnt/lustre/huangjunqin/NLPSpace/LLM/board_json_stage2_search/',
    # '/mnt/lustre/huangjunqin/NLPSpace/LLM/challenge-data/board_for_search/'
]
output_roots = [
    f'/mnt/lustre/huangjunqin/NLPSpace/LLM/REFINE_DT/trick_tools_v2/search_res/{KEY}/dev/',
    # f'/mnt/lustre/huangjunqin/NLPSpace/LLM/REFINE_DT/trick_tools_v2/search_res/{KEY}/board2/',
    # f'/mnt/lustre/huangjunqin/NLPSpace/LLM/REFINE_DT/trick_tools_v2/search_res/{KEY}/board/',
]

queries_file = ['challenge_ma']
############ hyper parameter

# load embedding model
model = SentenceTransformer('/mnt/lustre/share_data/huangjunqin/Embedding/HF_Model/bge-base-en-v1.5/').cuda()
origin_dataset = defaultdict()

print('start to load origin data')
with open(os.path.join(root, f'{dataset_name}.jsonl')) as f:
    for idx, line in enumerate(tqdm(f.readlines())):
        origin_dataset[idx] = json.loads(line.strip())

Embeddings = defaultdict()
Embeddings_idx = defaultdict()
embeddings_length = 0
print('start loading embeddings ...')
for i in tqdm(range(num_file)):
    with open(os.path.join(embedding_root, f'filter_long_en_p{i}.pkl'), 'rb') as pkl_file:
        embedding_dict = pickle.load(pkl_file)
        Embeddings[i] = torch.tensor(np.array(list(embedding_dict.values()))).cuda()
        Embeddings_idx[i] = list(embedding_dict.keys())
        embeddings_length += len(embedding_dict.keys())
assert embeddings_length == len(origin_dataset), 'embedding idx length != origin dataset length'

def search_topk(emb, topk=3):
    combined_results = []
    for part_idx, embeddings in Embeddings.items():
        similarities = F.cosine_similarity(emb.unsqueeze(0), embeddings, dim=1)
        topk_indices = torch.argsort(similarities, descending=True)[:topk]
        results = [(Embeddings_idx[part_idx][idx.item()], float(similarities[idx.item()]), origin_dataset[Embeddings_idx[part_idx][idx.item()]]) for idx in topk_indices]
        combined_results.extend(results)

    combined_results.sort(key=lambda x: x[1], reverse=True)
    topk_ids = [result for result in combined_results[:topk]]
    return topk_ids

## load candidate query
print('start to search ...')
all_results = defaultdict()
for query_root, output_root in zip(query_roots, output_roots):
    os.makedirs(output_root, exist_ok=True)
    for query_file in tqdm(queries_file):
        with open(os.path.join(query_root, f'{query_file}.json'), 'r') as f:
            queries = json.load(f)
            query_results = []
            for query in queries:
                emb = model.encode(query['text'], normalize_embeddings=True)
                res = search_topk(emb=torch.tensor(emb).cuda(), topk=TOPK)
                if KEY == 'input':
                    query_results.append({'question': query['text'], 'results': res})
                elif KEY == 'output':
                    query_results.append({'question': prefix_instruction + query['text'], 'results': res})
                else:
                    assert False, 'error'
                    
        with open(os.path.join(output_root, f'{query_file}.json'), 'w') as output_file:
            json.dump(query_results, output_file, indent=2)
