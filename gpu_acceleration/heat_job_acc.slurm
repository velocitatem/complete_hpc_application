#!/bin/bash
#SBATCH --job-name=heat_parallel_acc
#SBATCH --exclude=gpu-node[1-3]
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=heat_parallel_acc.out

module load gcc/9.3.0
module load openmpi/4.0.3
module load cuda/11.2

mpirun -np 2 ./heat_parallel_acc
