import torch
import os
import numpy as np
import pickle
import torch.multiprocessing as mp
import argparse
from tqdm import tqdm

'''
这里对S1中处理得到的各个part之间进行去重, 为了加快运算速度, 这里用使用多进程。
'''
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--embedding_path", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--thresh", required=True, type=float)
    parser.add_argument("--np", required=True, type=int)
    return parser.parse_args()

args = parse_args()

def compute_similarity_gpu(exist_emb, candidate_emb, candidate_idx, result_queue, rank, chunk_size):
    START_IDX, END_IDX = chunk_size * rank, chunk_size * (rank + 1)
    local_results = []
    print('-------------------------------------------')
    print(f'process {rank} starts process {START_IDX} to {END_IDX}')
    exist_emb_tensor = torch.tensor(np.array(exist_emb)).to(rank)
    candidate_emb_tensor = torch.tensor(np.array(candidate_emb)).to(rank)
    for c_idx, embedding in zip(candidate_idx[START_IDX: END_IDX], candidate_emb_tensor[START_IDX: END_IDX]):
        similarity_scores = torch.nn.functional.cosine_similarity(embedding.unsqueeze(0), exist_emb_tensor)
        if torch.max(similarity_scores) <= float(args.thresh):
            local_results.append(c_idx)
    
    # 将局部结果存入队列
    result_queue.extend(local_results)
    print(f'process {rank} done ...')

if __name__ == '__main__':
    with open(args.embedding_path, 'rb') as pkl_file:
        Embeddings = pickle.load(pkl_file)
    
    EXISTING_EMBEDDINGS = []
    EXISTING_IDX = []
    FILES = []

    for file_name in os.listdir(args.root):
        if str(file_name).endswith('.pkl'):
            FILES.append(file_name)
    
    FILES = sorted(FILES)
    for file_name in FILES:
        if len(EXISTING_IDX) == 0:
            # 读取 pkl 文件
            print(f'pre adding {file_name} ...')
            with open(os.path.join(args.root, file_name), 'rb') as pkl_file:
                for idx in list(pickle.load(pkl_file)):
                    EXISTING_EMBEDDINGS.append(Embeddings[idx])
                    EXISTING_IDX.append(idx)
            print(f'done to process {file_name}, and update results, exist embedding length: {len(EXISTING_IDX)}')
            for _ in range(10):
                print('-----------' * 10)
        else:
            with open(os.path.join(args.root, file_name), 'rb') as pkl_file:
                print(f'start to process {file_name} ...')
                candidate_emb, candidate_idx = [], []
                for idx in list(pickle.load(pkl_file)):
                    candidate_emb.append(Embeddings[idx])
                    candidate_idx.append(idx)
                
                # 使用 Manager 创建共享队列
                manager = mp.Manager()
                result_queue = manager.list()

                # 创建进程池
                processes = []

                # 启动多个进程进行相似度计算，每个进程使用不同的 GPU
                chunk_size = len(candidate_idx) // args.np + 1
                for rank in range(args.np):
                    p = mp.Process(target=compute_similarity_gpu, args=(EXISTING_EMBEDDINGS, candidate_emb, candidate_idx, result_queue, rank, chunk_size))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()

                # 更新 EXISTING_EMBEDDINGS 和 EXISTING_IDX
                EXISTING_IDX.extend(result_queue)
                EXISTING_EMBEDDINGS.extend([Embeddings[idx] for idx in result_queue])

                print(f'done to process {file_name}, and update results, exist embedding length: {len(EXISTING_IDX)}')
                for _ in range(10):
                    print('-----------' * 10)

    # 使用 pickle 保存 NumPy 数组为 pkl 文件
    with open(os.path.join(args.output_root, f'{args.thresh}_idx.pkl'), 'wb') as pkl_file:
        EXISTING_IDX = sorted(EXISTING_IDX) # 从小到大排序
        pickle.dump(EXISTING_IDX, pkl_file)