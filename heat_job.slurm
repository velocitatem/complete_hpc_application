#!/bin/bash

# This SLURM script is used to submit a parallel heat distribution simulation job to the cluster.

#SBATCH --job-name=heat_parallel  # Job name
#SBATCH --partition=cpubase_bycore_b1  # Partition name
#SBATCH --nodes=5  # Number of nodes
#SBATCH --ntasks-per-node=4  # Number of tasks per node
#SBATCH --cpus-per-task=1  # Number of CPUs per task
#SBATCH --time=00:30:00  # Time limit
#SBATCH --output=heat_parallel.out  # Standard output and error log

module load gcc/9.3.0  # Load GCC module
module load openmpi/4.0.3  # Load OpenMPI module

mpirun -np 12 ./heat_parallel  # Run the parallel heat distribution simulation
