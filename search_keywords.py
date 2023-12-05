import json
from tqdm import tqdm

KEYWORDS = ['summarize', 'Summarize', 'give a summary', 'a brief summary']

# load embedding model
rse_file = open('summary.jsonl', 'w') 
path = '/mnt/lustre/huangjunqin/NLPSpace/LLM/REFINE_DT/trick_tools_v2/filter_long_data/filter_en.jsonl'
with open(path, 'r') as f:
    for idx, line in tqdm(enumerate(f.readlines())):
        data = json.loads(line.strip())
        data.pop('input_ids')
        data.pop('split_ids')
        data.pop('labels')
        for keyword in KEYWORDS:
            # if keyword in data['instruction'] and (len(data['instruction']) + len(data['input']) > 1000):
            if keyword in data['instruction']:
                data['idx'] = idx
                rse_file.writelines(json.dumps(data, ensure_ascii=False) + '\n')