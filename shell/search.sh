srun --mpi=pmi2 -p Model-One --quotatype=auto -n1 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task 5 \
    python search_topk.py \