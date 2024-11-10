#!/bin/bash
#SBATCH --job-name=heat_parallel
#SBATCH --partition=cpubase_bycore_b1
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=heat_parallel.out

module load gcc/9.3.0
module load openmpi/4.0.3

mpirun -np 12 ./heat_parallel
