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

  #pragma omp parallel private(i, j, diff) shared(u, u_new, max_diff)
  for (iter = 0; iter < MAX_ITER; iter++) {
      max_diff = 0.0;

      for (i = 1; i < NX - 1; i++) {
          for (j = 1; j < NY - 1; j++) {
              u_new[i][j] = 0.25 * (u[i+1][j] + u[i-1][j]
                                   + u[i][j+1] + u[i][j-1]);
              diff = fabs(u_new[i][j] - u[i][j]);
              if (diff > max_diff) {
                  max_diff = diff;
              }
          }
      }

      // Update u
      for (i = 1; i < NX - 1; i++) {
          for (j = 1; j < NY - 1; j++) {
              u[i][j] = u_new[i][j];
          }
      }

      // Check for convergence
      if (max_diff < TOLERANCE) {
          printf("Converged after %d iterations.\n", iter);
          break;
      }
  }

  for (i = 0; i < NX; i++) {
     for (j = 0; j < NY; j++) {
         printf("%f ", u[i][j]);
     }
     printf("\n");
  }
  MPI_Finalize();
  return 0;
}
