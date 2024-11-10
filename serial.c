#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define NX 100
#define NY 100
#define MAX_ITER 100
#define TOLERANCE 1e-6

int main() {
   double u[NX][NY], u_new[NX][NY];
   int i, j, iter;
   double diff, max_diff;

   double start_time = clock();

   // Initialize the grid
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

   double end_time = clock();
   double elapsed_time = (end_time - start_time) / CLOCKS_PER_SEC;
   printf("Elapsed time: %f seconds\n", elapsed_time);

   for (i = 0; i < NX; i++) {
      for (j = 0; j < NY; j++) {
          printf("%6.2f ", u[i][j]);
      }
      printf("\n");
   }

   return 0;
}
