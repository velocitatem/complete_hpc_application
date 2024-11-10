#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <openacc.h>

#define NX 100
#define NY 100
#define MAX_ITER 100
#define TOLERANCE 1e-6

int main() {
    double u[NX][NY], u_new[NX][NY];
    int i, j, iter;
    double diff, max_diff;

    MPI_Init(NULL, NULL);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start_time = MPI_Wtime();

    if (NX % size != 0) {
        if (rank == 0) {
            printf("NX must be divisible by the number of processes.\n");
        }
        MPI_Finalize();
        return 1;
    }
    int row_chunks = NX / size; // Number of rows per process

    // Initialize the grid in rank 0
    if (rank == 0) {
        for (i = 0; i < NX; i++) {
            for (j = 0; j < NY; j++) {
                u[i][j] = 0.0;
                if (i == 0 || i == NX - 1 || j == 0 || j == NY - 1) {
                    u[i][j] = 100.0; // Boundary conditions
                }
            }
        }
    }

    // Allocate memory for local grids, including halo rows
    double *u_local = (double *)malloc((row_chunks + 2) * NY * sizeof(double)); // +2 for halo rows
    double *u_new_local = (double *)malloc((row_chunks + 2) * NY * sizeof(double));

    // Initialize local grids
    for (i = 0; i < row_chunks + 2; i++) {
        for (j = 0; j < NY; j++) {
            u_local[i * NY + j] = 0.0;
            u_new_local[i * NY + j] = 0.0;
        }
    }

    // Scatter the grid data (without halo regions)
    MPI_Scatter(&u[0][0], row_chunks * NY, MPI_DOUBLE, &u_local[NY], row_chunks * NY, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Ensure top and bottom boundaries are maintained by relevant processes
    if (rank == 0) {
        for (j = 0; j < NY; j++) {
            u_local[j] = 100.0; // Top boundary
        }
    }
    if (rank == size - 1) {
        for (j = 0; j < NY; j++) {
            u_local[(row_chunks + 1) * NY + j] = 100.0; // Bottom boundary
        }
    }

    // Main iteration loop
    #pragma acc data copyin(u[0:NX][0:NY]) copyout(u_new[0:NX][0:NY])
    for (iter = 0; iter < MAX_ITER; iter++) {
        // Halo exchange
        if (rank > 0) {
            MPI_Sendrecv(&u_local[NY], NY, MPI_DOUBLE, rank - 1, 0,
                         &u_local[0], NY, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank < size - 1) {
            MPI_Sendrecv(&u_local[row_chunks * NY], NY, MPI_DOUBLE, rank + 1, 0,
                         &u_local[(row_chunks + 1) * NY], NY, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // Compute new values using OpenACC (excluding boundary rows)
        max_diff = 0.0;
        #pragma acc parallel loop collapse(2) private(i, j, diff) reduction(max:max_diff)
        for (i = 1; i <= row_chunks; i++) {
            for (j = 1; j < NY - 1; j++) {
                if (rank == 0 && i == 1) {
                  u_new_local[i * NY + j] = 100.0; // Top boundary
                  continue;
                }
                if (rank == size - 1 && i == row_chunks) {
                  u_new_local[i * NY + j] = 100.0; // Bottom boundary
                  continue;
                }
                u_new_local[i * NY + j] = 0.25 * (u_local[(i + 1) * NY + j] + u_local[(i - 1) * NY + j]
                                                 + u_local[i * NY + j + 1] + u_local[i * NY + j - 1]);
                diff = fabs(u_new_local[i * NY + j] - u_local[i * NY + j]);
                if (diff > max_diff) {
                    max_diff = diff;
                }
            }
        }

        // Update local grid with new values (excluding boundary rows)
        for (i = 1; i <= row_chunks; i++) {
            for (j = 1; j < NY - 1; j++) {
                u_local[i * NY + j] = u_new_local[i * NY + j];
            }
        }

        // Ensure boundary values remain fixed for processes handling boundaries
        if (rank == 0) {
            for (j = 0; j < NY; j++) {
                u_local[j] = 100.0; // Top boundary
            }
        }
        if (rank == size - 1) {
            for (j = 0; j < NY; j++) {
                u_local[(row_chunks + 1) * NY + j] = 100.0; // Bottom boundary
            }
        }

        // Check for convergence
        double global_max_diff;
        MPI_Allreduce(&max_diff, &global_max_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (global_max_diff < TOLERANCE) {
            if (rank == 0) {
                printf("Converged after %d iterations.\n", iter);
            }
            break;
        }
    }

    // Gather the grid data back to rank 0
    MPI_Gather(&u_local[NY], row_chunks * NY, MPI_DOUBLE, &u[0][0], row_chunks * NY, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double end_time = MPI_Wtime();
    if (rank == 0) {
        printf("Execution time: %f seconds\n", end_time - start_time);
    }

    if (rank == 0) {
        // Print the final grid
        for (i = 0; i < NX; i++) {
            for (j = 0; j < NY; j++) {
                printf("%6.2f ", u[i][j]);
            }
            printf("\n");
        }
    }

    // Free allocated memory
    free(u_new_local);
    free(u_local);

    MPI_Finalize();
    return 0;
}
