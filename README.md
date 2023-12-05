# 数据处理流程
数据处理流程总共包含多个步骤, step0为data-jucier官方code
这里的readme以英文为例，中文可以使用相同的流程进行处理

## STEP 0: data-jucier 基础过滤
使用data-jucier的官方代码对英文文本进行筛选去重  
将data-juicer中的配置文件<KIT_DIR>/data-juicer/configs/data_juicer_recipes/alpaca_cot/alpaca-cot-en-refine.yaml修改为raw_data/raw_data_en.jsonl,

然后执行如下命令对提供的原始数据集进行改良：其中<KIT_DIR>表示竞赛套件所在目录的路径
cd <KIT_DIR>/data-juice
python tools/process_data.py --config configs/data_juicer_recipes/alpaca_cot/alpaca-cot-en-refine.yaml

以上操作会得到alpaca-cot-en-refine.jsonl

## STEP 1: 过滤长文本 
对alpaca-cot-en-refine.jsonl进行按token过滤，对于大于1024 token的样本，全部过滤掉。
过滤的脚本为 filter_long_data.py。过滤之后得到数据 filter_long_en.jsonl，约300w数据

## STEP 2: 向量召回 + 关键字召回
这里以所有的验证集为召回种子数据，通过将训练数据和种子数据进行向量化（英文采用bge-base-en-v1.5, 中文采用piccolo-base-zh）,
中文召回TOP5，英文召回TOP70.
向量召回需要用到的脚本为:  
1. shell/encode_multi.sh, 对大规模数据进行并行向量化。
2. shell/search.sh, 对种子数据进行向量召回
关键字召回需要用到的脚本为:
1. search_keywords.py

将召回结果通过search_res_to_train.py进行整合。整合时，默认score高的在前面。

## STEP 3: 向量去重
使用embedding模型对召回的数据进行去重，英文去重thresh为0.75，中文为0.7。
相似的向量之间，保留召回score最高的那个。
需要用到的脚本为:
1. shell/dedup_s1.sh
2. shell/dedup_s2.sh
脚本均已做过并行加速。

## STEP 4: 后处理
使用get_train_dataset.py对处理结果进行筛选。筛选10M token。