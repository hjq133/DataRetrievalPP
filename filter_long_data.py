import transformers
import torch
from datasets import load_dataset
from datasets import Dataset
from typing import Dict
import copy

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "en": {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    )},
    "zh": {
    "prompt_input": (
        "以下是描述任务的指示，配有提供进一步上下文的输入，编写一个适当的回应完成请求\n\n"
        "### 指示：\n{instruction}\n\n### 输入：\n{input}\n\n### 回应："
    ),
    "prompt_no_input": (
        "以下是描述任务的指示，编写一个适当的回应完成请求\n\n"
        "### 指示：\n{instruction}\n\n### 回应："
    )}
}


def update_token_id(model, tokenizer):
    # To solve the bug of llama config
    for name in ['bos', 'eos', 'pad', 'unk']:
        
        token_id_name = '_'.join([name, 'token_id'])
        token_name = '_'.join([name, 'token'])

        token_id = getattr(tokenizer,  token_id_name)
        if token_id is None:
            token_str = getattr(tokenizer,  token_name)
            token_id = tokenizer.encode(token_str, add_special_tokens=False)[0]

        setattr(tokenizer, token_id_name, token_id)
        setattr(model.config, token_id_name, token_id)


def format_data(lang: str, dataset: Dataset, num_proc:int = 1):
    prompt_input, prompt_no_input = PROMPT_DICT[lang]["prompt_input"], PROMPT_DICT[lang]["prompt_no_input"]
    def add_prompt(example):
        if "instruction" in example and "output" in example:
            example["target"] = example["output"]
            if example.get("input", "") != "":
                example["source"] = prompt_input.format_map(example)
            else:
                example["source"] = prompt_no_input.format_map(example)
            return example
        else:
            raise RuntimeError(f"{example}")
    return dataset.map(add_prompt, num_proc=num_proc)


def preprocess(
    format_dataset: Dataset,
    tokenizer: transformers.PreTrainedTokenizer,
    num_proc:int = 1,
) -> Dict:
 
    def _tokenize_fn(example):
        """Tokenize example"""
        example["source"] = tokenizer(example["source"], return_tensors="pt", padding="longest",
                                      max_length=tokenizer.model_max_length, truncation=True,
                                      add_special_tokens=False)
        example["target"] = tokenizer(example["target"], return_tensors="pt", padding="longest",
                                      max_length=tokenizer.model_max_length, truncation=True,
                                      add_special_tokens=False)
                            
        source_input_id = source_label = example["source"].input_ids[0]          
        target_input_id = target_label = torch.cat([example["target"].input_ids[0], torch.tensor([tokenizer.eos_token_id])])
        input_id = torch.cat([source_input_id, target_input_id])
        label = copy.deepcopy(input_id)
        label[:len(source_input_id)] = IGNORE_INDEX

        example["input_ids"] = input_id
        example["labels"] = label
        example["split_ids"] = len(source_input_id)
        return example

    """Preprocess the data by tokenizing."""
    processed_dataset = format_dataset.map(_tokenize_fn, remove_columns=["source", "target"],num_proc=num_proc)
    return processed_dataset

def filter_strategy(example):
    '''
    filter掉max-length大于1024的, 然后filter掉那些文本里带http的 
    '''
    if len(example["input_ids"]) > tokenizer.model_max_length:
        return False
    if 'http' in example['text']:
        return False
    else:
        return True
    
    
if __name__  == '__main__':
    # data_path = '/mnt/lustre/huangjunqin/NLPSpace/LLM/REFINE_DT/refine_data_en/alpaca-cot-en-refine.jsonl'
    data_path = '/mnt/lustre/huangjunqin/NLPSpace/LLM/REFINE_DT/refine_data_zh/alpaca-cot-zh-refine.jsonl'
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        '/mnt/lustre/share_data/huangjunqin/LLM/model/Baichuan2-7B-Base/',
        model_max_length=1024,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True
    )
    dataset = load_dataset('json', data_files=data_path, split="train")
    format_dataset = format_data('en', dataset, num_proc = 50)
    dataset  = preprocess(format_dataset, tokenizer, num_proc = 50)
    dataset = dataset.filter(filter_strategy, num_proc=50)
    dataset.to_json('filter_long_zh.jsonl', force_ascii=False)