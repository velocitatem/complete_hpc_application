#!/bin/bash
#SBATCH --job-name=heat_parallel_acc
#SBATCH --nodes=2
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=heat_parallel_acc.out

module load gcc
module load openmpi
module load nvhpc

mpirun -np 2 ./heat_parallel_acc
