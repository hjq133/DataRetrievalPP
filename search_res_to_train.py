import json
import os
from collections import defaultdict
from tqdm import tqdm

topk = 1000
queries_file = ['challenge_ma', 'challenge_mc', 'challenge_mc_ctx', 'challenge_mc_massive', 'challenge_mc_long', 'challenge_qmsumm', 'challenge_summ']
search_strategy = ['input/dev', 'input/board', 'input/board2', 'output/dev', 'output/board', 'output/board2']
search_root = '/mnt/lustre/huangjunqin/NLPSpace/LLM/REFINE_DT/trick_tools_v2/search_res/'

DATASET = defaultdict()
cnt = 0
print('process search results ...')
all_results = defaultdict()
for strategy in search_strategy:
    for query_file in tqdm(queries_file):
        with open(os.path.join(search_root, strategy, f'{query_file}.json'), 'r') as f:
            datas = json.load(f)
            for data_topk in datas:
                for data in data_topk['results'][:topk]:
                    dataset_idx = data[0]
                    dataset_name = data[2]['meta']['Dataset']
                    data[2]['idx'] = dataset_idx
                    data[2]['search_score'] = data[1]
                    # 始终只保留score最大的
                    if dataset_idx not in DATASET.keys():
                        DATASET[dataset_idx] = data[2]
                    else:
                        if data[2]['search_score'] > DATASET[dataset_idx]['search_score']:
                            DATASET[dataset_idx] = data[2]
                    cnt += 1

print('origin search cnt is ', cnt)
print('start to sort data ...')
# 按照 search_score 进行排序，从大到小
# 按照 search_score 进行排序，从大到小
sorted_dataset = sorted(DATASET.values(), key=lambda x: x['search_score'], reverse=True)

# 保存排序后的结果为 JSON Lines 格式
with open('res_en_v7.jsonl', 'w') as output_file:
    for item in sorted_dataset:
        json.dump(item, output_file)
        output_file.write('\n')
