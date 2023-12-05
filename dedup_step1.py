import pickle
import torch
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--thresh", required=True, type=float)
    parser.add_argument("--rank_num", required=True, type=int)
    return parser.parse_args()

args = parse_args()
print('start filtering with thresh: ', args.thresh)

# 检查CUDA是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_ROOT = args.output_root
os.makedirs(OUTPUT_ROOT, exist_ok=True)
pkl_file_path = os.path.join(args.root, f'part_{args.rank_num}.pkl' )

# 读取 pkl 文件

with open(pkl_file_path, 'rb') as pkl_file:
    embedding_dict = pickle.load(pkl_file)
    embeddings_tensor = torch.tensor(np.array(list(embedding_dict.values()))).to(device)
    embeddings_idx = list(embedding_dict.keys())

# 转换嵌入向量为PyTorch张量并将其移动到GPU
existing_idx = []
existing_embeddings = []
similarity_scores_list = []

print(f'embeddings length: {len(embeddings_tensor)}')
print(f'process {args.rank_num} start to filter embeddings from {pkl_file_path}')

for idx, embedding in tqdm(zip(embeddings_idx, embeddings_tensor)):
    if not existing_idx:
        existing_idx.append(idx)
        existing_embeddings.append(embedding)
    else:
        # 计算与现有指令的相似度分数
        similarity_scores = torch.nn.functional.cosine_similarity(embedding.unsqueeze(0), torch.stack(existing_embeddings))

        # 如果新指令与现有指令足够不相似，将其添加到 final_data 中
        similarity_scores_list.append(torch.max(similarity_scores).cpu().numpy())
        if torch.max(similarity_scores) <= args.thresh:
            existing_idx.append(idx)
            existing_embeddings.append(embedding)

print('origin idx length is : ', len(embeddings_tensor))
print('exsiting idx length is : ', len(existing_idx))
print('keep ratio: ', float(len(existing_idx)) / float(len(embeddings_tensor)))
# 使用 pickle 保存 NumPy 数组为 pkl 文件
with open(os.path.join(OUTPUT_ROOT, f'{args.rank_num}.pkl'), 'wb') as pkl_file:
    pickle.dump(existing_idx, pkl_file)
    
# 绘制相似度分布图
plt.hist(similarity_scores_list, bins=100, color='blue', edgecolor='black')
plt.title('Cosine Similarity Distribution')
plt.xlabel('Cosine Similarity')
plt.ylabel('Frequency')
plt.savefig(os.path.join(OUTPUT_ROOT, f'part_{args.rank_num}.png'))