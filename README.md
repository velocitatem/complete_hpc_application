# complete_hpc_application

## OpenMPI Parallelization

This project uses OpenMPI to parallelize the heat distribution simulation. The parallelized version of the code is implemented in `heat_parallel.c`.

### Compilation
module load gcc/9.3.0
module load openmpi/4.0.3
To compile the parallelized code, use the following command:

```sh
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
