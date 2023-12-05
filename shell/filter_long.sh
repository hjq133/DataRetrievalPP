srun --mpi=pmi2 -p Model-One --quotatype=auto -n1 --gres=gpu:8 --ntasks-per-node=1 --cpus-per-task 40 \
    python filter_long_data.py
