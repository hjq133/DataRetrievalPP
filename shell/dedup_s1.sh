## Hyper Parameter Start
Thresh=0.7
ROOT=/mnt/lustre/huangjunqin/NLPSpace/LLM/REFINE_DT/trick_tools_v2/final_output/search_v8/before_dedup_emb_zh
OUTPUT_ROOT=/mnt/lustre/huangjunqin/NLPSpace/LLM/REFINE_DT/trick_tools_v2/final_output/search_v8/dedup_zh/$Thresh
LOG_ROOT=/mnt/lustre/huangjunqin/NLPSpace/LLM/REFINE_DT/trick_tools_v2/final_output/search_v8/dedup_zh_logs/$Thresh
PROCESS_NUM=4
## Hyper Parameter End

mkdir -p $LOG_ROOT
mkdir -p $OUTPUT_ROOT

for ((i=0;i<$PROCESS_NUM;i+=1)); do
    (
        dataset=$DATASET
        echo "[DATASET] $dataset part $i submit"
        srun --mpi=pmi2 -p Model-One --quotatype=auto -n1 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task 5 --job-name=$dataset_$i \
            python dedup_step1.py \
                --root=$ROOT \
                --thresh=$Thresh \
                --rank_num=$i \
                --output_root=$OUTPUT_ROOT > $LOG_ROOT/$dataset_$i.$Thresh.log 2>&1
    )& 
    sleep 2
done

echo "[Submit Done]"