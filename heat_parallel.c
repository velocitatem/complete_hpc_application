#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

#define NX 10
#define NY 10
#define MAX_ITER 3
#define TOLERANCE 1e-6

int main() {
    double u[NX][NY], u_new[NX][NY];
    int i, j, iter;
    double diff, max_diff;

    MPI_Init(NULL, NULL);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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
      // Initialize the grid with boundary values
      for (i = 0; i < NX; i++) {
        for (j = 0; j < NY; j++) {
          u[i][j] = 0.0;
          if (i == 0 || i == NX - 1 || j == 0 || j == NY - 1) {
            u[i][j] = 100.0;
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

    // Main iteration loop
    for (iter = 0; iter < MAX_ITER; iter++) {

        if (rank > 0) { // Send/Recv upper halo
          MPI_Sendrecv(&u_local[NY], NY, MPI_DOUBLE, rank - 1, 0, &u_local[0], NY,
                      MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD,
                      MPI_STATUS_IGNORE);
        }
        if (rank < size - 1) { // Send/Recv lower halo
          MPI_Sendrecv(&u_local[row_chunks * NY], NY, MPI_DOUBLE, rank + 1, 0,
                      &u_local[(row_chunks + 1) * NY], NY, MPI_DOUBLE, rank + 1,
                      0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // Compute new values using OpenMP
        max_diff = 0.0;
        #pragma omp parallel for private(i, j, diff) reduction(max : max_diff)
        for (i = 1; i <= row_chunks;
             i++) { // Ensure it does not include halo/boundary rows
          for (j = 1; j < NY - 1; j++) {
            u_new_local[i * NY + j] =
                0.25 * (u_local[(i + 1) * NY + j] + u_local[(i - 1) * NY + j] +
                        u_local[i * NY + j + 1] + u_local[i * NY + j - 1]);
            diff = fabs(u_new_local[i * NY + j] - u_local[i * NY + j]);
            if (diff > max_diff) {
              max_diff = diff;
            }
          }
        }

        // Update local grid with new values
        for (i = 1; i <= row_chunks; i++) {
            for (j = 1; j < NY - 1; j++) {
                u_local[i * NY + j] = u_new_local[i * NY + j];
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
    MPI_Gather(&u_local[NY], row_chunks * NY, MPI_DOUBLE, &u[0][0],
               row_chunks * NY, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Print the final grid
        printf("All done\n");
        for (i = 0; i < NX; i++) {
            printf("%d: ", i);
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
