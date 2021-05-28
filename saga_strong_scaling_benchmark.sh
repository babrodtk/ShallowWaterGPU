#!/bin/bash

# one node: 1-4 tasks/GPUs
sbatch --partition=accel --gres=gpu:1 --nodes=1 --ntasks-per-node=1 saga_strong_scaling_benchmark.job
sbatch --partition=accel --gres=gpu:2 --nodes=1 --ntasks-per-node=2 saga_strong_scaling_benchmark.job
sbatch --partition=accel --gres=gpu:3 --nodes=1 --ntasks-per-node=3 saga_strong_scaling_benchmark.job
sbatch --partition=accel --gres=gpu:4 --nodes=1 --ntasks-per-node=4 saga_strong_scaling_benchmark.job

# 2-4 nodes: 4 tasks/GPUs per node
sbatch --partition=accel --gres=gpu:4 --nodes=2 --ntasks-per-node=4 saga_strong_scaling_benchmark.job
sbatch --partition=accel --gres=gpu:4 --nodes=3 --ntasks-per-node=4 saga_strong_scaling_benchmark.job
sbatch --partition=accel --gres=gpu:4 --nodes=4 --ntasks-per-node=4 saga_strong_scaling_benchmark.job

