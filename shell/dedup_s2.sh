## Hyper Parameter Start
Thresh=0.7
ROOT=/mnt/lustre/huangjunqin/NLPSpace/LLM/REFINE_DT/trick_tools_v2/final_output/search_v8/dedup_zh/$Thresh
OUTPUT_ROOT=$ROOT/MERGE/
LOG_ROOT=$OUTPUT_ROOT
GPU_NUM=4
EmbeddingPath=/mnt/lustre/huangjunqin/NLPSpace/LLM/REFINE_DT/trick_tools_v2/final_output/search_v8/before_dedup_emb_zh/keep_emb.pkl
## Hyper Parameter End

mkdir -p $LOG_ROOT
mkdir -p $OUTPUT_ROOT
echo "START"
srun -p Model --quotatype=auto -n1 --gres=gpu:$GPU_NUM --ntasks-per-node=1 --cpus-per-task 20 --job-name=merge \
    python dedup_step2.py \
        --root=$ROOT \
        --embedding_path=$EmbeddingPath \
        --thresh=$Thresh \
        --np=$GPU_NUM \
        --output_root=$OUTPUT_ROOT > $LOG_ROOT/merge.$Thresh.log 2>&1 \
        &

echo "[Submit Done]"