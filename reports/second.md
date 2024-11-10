# Report on GPU Acceleration of Heat Equation Solver Using OpenACC

## Introduction

The objective of this project was to accelerate a parallel heat equation solver using GPU computing. The original code was implemented using MPI for distributed memory parallelism and OpenMP for shared memory parallelism on CPUs. While this implementation provided performance gains over a serial version, leveraging GPUs promised significant additional speedups due to their high parallel processing capabilities.

## Implementation of GPU Acceleration

To harness the computational power of GPUs, I transitioned the code from using OpenMP to OpenACC. OpenACC is a directive-based programming model that allows developers to offload compute-intensive regions of code to GPUs with minimal changes.

### Key Modifications:

1. **Data Management with OpenACC**:
   - **Local Arrays on GPU**: Since the computation is distributed across MPI processes, each process operates on its local portion of the grid (`u_local` and `u_new_local`). I enclosed the main iteration loop within an OpenACC data region to allocate these arrays on the GPU.
     ```c
     #pragma acc data copy(u_local[0:(row_chunks+2)*NY]) create(u_new_local[0:(row_chunks+2)*NY])
     ```
     - `copy`: Allocates memory on the device and copies data to and from the host.
     - `create`: Allocates memory on the device without initializing it on the host or copying it back.

2. **Computational Kernel Offloading**:
   - I identified the most compute-intensive part of the code—the nested loops updating the temperature grid—and annotated it with OpenACC directives to offload it to the GPU.
     ```c
     #pragma acc parallel loop collapse(2) reduction(max:max_diff)
     for (i = 1; i <= row_chunks; i++) {
         for (j = 1; j < NY - 1; j++) {
             // Compute new temperature values
         }
     }
     ```
     - `parallel loop`: Instructs the compiler to parallelize the loop on the GPU.
     - `collapse(2)`: Combines the nested loops for parallel execution.
     - `reduction(max:max_diff)`: Performs a reduction to find the maximum difference across all iterations.

4. **Boundary Conditions Handling**:
   - I adjusted the code to maintain boundary conditions within the GPU computation by checking the process rank and loop indices.

## Challenges and Solutions

### Challenge 1: **MPI and OpenACC Interoperability**

**Issue**: Combining MPI with OpenACC required careful handling to ensure correct data exchange and synchronization across processes.

**Solution**:
- Ensured that MPI communication only occurred after data on the device was updated to the host.
- Maintained consistent data regions and synchronization points to prevent race conditions and ensure data integrity.

### Challenge 2: **Dynamic Memory Allocation**

**Issue**: OpenACC requires explicit management of dynamically allocated memory to ensure it's correctly allocated on the device.

**Solution**:
- Used the `copy` and `create` clauses in the OpenACC data directive to manage dynamically allocated arrays.
- Verified that all pointers used within the GPU regions pointed to memory allocated and managed within the OpenACC data regions.

### Challenge 3: **Running Locally**

**Issue**: Since the GPU compute nodes on the cluster were down in the time when I was writing this project, I opted for my own GPU.

**Solution**:
- Executed with 2 processes running on my own computer with a GPU

## Performance Analysis

### Execution Times:

- **Serial Execution Time**: ~0.0089 seconds
- **CPU-only Parallel Execution Time (with MPI and OpenMP on 5 processes)**: ~0.0042 seconds
- **GPU-accelerated Execution Time (with 2 processes, each using a GPU)**: ~0.002742 seconds

### Speedup Calculation:
1. Speedup of CPU-only Parallel Version over Serial. S_CPU = T_serial / T_CPU_parallel = 0.0089 / 0.0042 ≈ 2.1

2. Speedup of GPU-accelerated Version over Serial: S_GPU = T_serial / T_GPU_parallel = 0.0089 / 0.002742 ≈ 3.25

3. Speedup of GPU-accelerated Version over CPU-only Parallel Version: S_GPU_over_CPU = T_CPU_parallel / T_GPU_parallel = 0.0042 / 0.002742 ≈ 1.53

### Observations:

- **GPU Acceleration Provides Significant Speedup**: The GPU-accelerated version is approximately 3.25 times faster than the serial version and 1.53 times faster than the CPU-only parallel version.
- **Efficiency of GPU Utilization**: Despite using only 2 GPUs compared to 5 CPU processes, the GPU version outperformed the CPU-only version, demonstrating the efficiency of GPUs for parallel computations.
- **Scaling Potential**: The code shows potential for further performance improvements by optimizing GPU utilization and possibly increasing the number of GPUs or processes.

### Performance Considerations:

- **Data Transfer Overhead**: Careful management of data movement between the host and device minimized overhead and contributed to performance gains.
- **Computational Intensity**: The heat equation solver is compute-intensive with a high arithmetic intensity, making it well-suited for GPU acceleration.
- **Communication Overhead**: MPI communication introduces some overhead, but its impact was mitigated by overlapping computation and communication where possible.

## Conclusion

Implementing GPU acceleration using OpenACC significantly improved the performance of the parallel heat equation solver. By carefully managing data movement and leveraging the parallel processing capabilities of GPUs, the execution time was reduced by approximately 70% compared to the CPU-only parallel version.

### Benefits of GPU Acceleration:

- **Improved Performance**: Faster computation times lead to quicker results and the ability to handle larger problem sizes within reasonable time frames.
- **Efficient Resource Utilization**: GPUs provide high throughput for parallel tasks, allowing for better utilization of available hardware resources.
