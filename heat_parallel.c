#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <openacc.h>
#include <mpi.h>

#define NX 500
#define NY 500
#define MAX_ITER 1000
#define TOLERANCE 1e-6

int main() {
  MPI_Init(NULL, NULL);

  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int rows_per_proc = NX / world_size;
  int start_row = world_rank * rows_per_proc;
  int end_row = (world_rank + 1) * rows_per_proc;

  double u[NX][NY], u_new[NX][NY];
  int i, j, iter;
  double diff, max_diff;

  // Initialize the grid
  #pragma omp parallel for private(i, j)
  for (i = 0; i < NX; i++) {
      for (j = 0; j < NY; j++) {
          u[i][j] = 0.0;
          if (i == 0 || i == NX - 1 || j == 0 || j == NY - 1) {
              u[i][j] = 100.0; // Boundary conditions
          }
      }
  }

  // Iterative solver
  for (iter = 0; iter < MAX_ITER; iter++) {
      max_diff = 0.0;

      for (i = start_row; i < end_row; i++) {
          for (j = 1; j < NY - 1; j++) {
              u_new[i][j] = 0.25 * (u[i+1][j] + u[i-1][j]
                                   + u[i][j+1] + u[i][j-1]);
              diff = fabs(u_new[i][j] - u[i][j]);
              if (diff > max_diff) {
                  max_diff = diff;
              }
          }
      }

      // Communicate boundary rows with neighboring processes
      if (world_rank > 0) {
          MPI_Send(&u_new[start_row][0], NY, MPI_DOUBLE, world_rank - 1, 0, MPI_COMM_WORLD);
          MPI_Recv(&u[start_row - 1][0], NY, MPI_DOUBLE, world_rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
      if (world_rank < world_size - 1) {
          MPI_Send(&u_new[end_row - 1][0], NY, MPI_DOUBLE, world_rank + 1, 0, MPI_COMM_WORLD);
          MPI_Recv(&u[end_row][0], NY, MPI_DOUBLE, world_rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }

      // Update u
      for (i = start_row; i < end_row; i++) {
          for (j = 1; j < NY - 1; j++) {
              u[i][j] = u_new[i][j];
          }
      }

      // Check for convergence
      MPI_Allreduce(&max_diff, &diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      if (diff < TOLERANCE) {
          if (world_rank == 0) {
              printf("Converged after %d iterations.\n", iter);
          }
          break;
      }
  }

  // Gather results from all processes
  MPI_Gather(&u[start_row][0], rows_per_proc * NY, MPI_DOUBLE, &u[0][0], rows_per_proc * NY, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (world_rank == 0) {
      for (i = 0; i < NX; i++) {
          for (j = 0; j < NY; j++) {
              printf("%f ", u[i][j]);
          }
          printf("\n");
      }
  }

  MPI_Finalize();
  return 0;
}
