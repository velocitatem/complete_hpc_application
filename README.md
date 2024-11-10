# complete_hpc_application

## Purpose of the Project

The purpose of this project is to simulate heat distribution using both serial and parallel implementations. The parallel implementation uses OpenMPI and OpenMP to distribute the computation across multiple processes and threads, respectively. The project aims to compare the performance and results of the serial and parallel implementations.

## Structure of the Repository

- `compare.ipynb`: Jupyter notebook to compare the results of the parallel and serial implementations.
- `heat_job.slurm`: SLURM script to submit the parallel heat distribution simulation job to a cluster.
- `heat_parallel.c`: Parallel implementation of the heat distribution simulation using OpenMPI and OpenMP.
- `serial.c`: Serial implementation of the heat distribution simulation.
- `README.md`: This file, providing an overview and instructions for the project.

## OpenMPI Parallelization

This project uses OpenMPI to parallelize the heat distribution simulation. The parallelized version of the code is implemented in `heat_parallel.c`.

### Compilation

To compile the parallelized code, use the following commands:

```sh
module load gcc/9.3.0
module load openmpi/4.0.3
mpicc -o heat_parallel heat_parallel.c -fopenmp -fopenacc
```

### Running the Code

To run the parallelized code, use the following command:

```sh
mpirun -np <number_of_processes> ./heat_parallel
```

For example, to run the code with 12 processes, use:

```sh
mpirun -np 12 ./heat_parallel
```

## Usage of the `compare.ipynb` Notebook

The `compare.ipynb` notebook is used to compare the results of the parallel and serial implementations of the heat distribution simulation. It reads the output files generated by both implementations, checks if the results are close, and visualizes the heat distributions.

To use the notebook, follow these steps:

1. Run the parallel and serial implementations to generate the output files (`parallel.out` and `serial.out`).
2. Open the `compare.ipynb` notebook in Jupyter.
3. Execute the cells in the notebook to compare and visualize the results.

## SLURM Script

The `heat_job.slurm` script is used to submit the parallel heat distribution simulation job to a cluster. It specifies the job name, partition, number of nodes, number of tasks per node, number of CPUs per task, time limit, and output file. The script also loads the necessary modules and runs the parallel heat distribution simulation using `mpirun`.

To submit the job, use the following command:

```sh
sbatch heat_job.slurm
```

## Comparing Results

You can compare the results of the parallel and serial implementations using the following commands or the `compare.ipynb` notebook:

```sh
mpirun -np 5 heat_parallel >> parallel.out
./serial >> serial.out
diff serial.out parallel.out
```

The time it took for the MPI and OpenMP code to run is ~2.1 times faster when running on 5 or more compute nodes with CPUs.


## GPUs
