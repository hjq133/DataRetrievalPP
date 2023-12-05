dataset=filter_long_en
ROOT='/mnt/lustre/huangjunqin/NLPSpace/LLM/REFINE_DT/trick_tools_v2/filter_long_data/'
OUTPUT_ROOT='encode_emb_output'
GPU_NUM=18
DATASET_NUM=3096532
mkdir -p logs_input

for ((i=1; i<3;i+=1)); do
    (
        echo "[DATASET] $dataset Part $i SUBMIT"
        srun --mpi=pmi2 -p Model-One --quotatype=auto -n1 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task 5 \
            python encode_emb_output.py \
                --dataset_name=$dataset \
                --root=$ROOT \
                --rank=$i \
                --world_size=$GPU_NUM \
                --dataset_num=$DATASET_NUM \
                --output_root=$OUTPUT_ROOT > logs_input/$dataset.part.$i.log 2>&1
    )& 
    sleep 2
done