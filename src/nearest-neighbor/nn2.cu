#include <iostream>
#include <stdio.h>

#include "nn2.hh"

#define cudaCheckError() {                                                   \
    cudaError_t e=cudaGetLastError();                                        \
    if(e!=cudaSuccess) {                                                     \
        printf("Cuda failure %s:%d: '%s'\n", __FILE__ , __LINE__,            \
               cudaGetErrorString(e));                                       \
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
}

namespace irgpu {


__global__ void l2_sq(double *mat1, double *mat2, double *l2_sq,
                      int M, int N, int P) {

    int tile_width = blockDim.x;  // square tile

    // Need to split the shared buffer to use two.
    extern __shared__ double tiles[];
    double *tile1 = (double*) tiles;
    double *tile2 = (double*) &tiles[tile_width * tile_width];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Indexes in the grid
    int row = by*blockDim.y + ty;
    int col = bx*blockDim.x + tx;

    double p_sum = 0;     
    for (int k = 0; k < ceil(N / (float) tile_width); k++) {

        // Load into shared memory and check boundaries
        if (row < M && (tx + k*tile_width) < N) {
            tile1[ty*blockDim.y + tx] = mat1[row*N + tx + k*tile_width];    // [row][tx + patch_shift]
        }
        if ((ty + k * tile_width) < N && col < P) {
            tile2[ty*blockDim.y + tx] = mat2[(ty + k*tile_width)*P + col];  // [ty + patch_shift][col]
        }
        //printf("%d %d %f %f %f %f\n", p1, p2, tile1[ty*blockDim.y + tx], tile2[ty*blockDim.y + tx],
        //    mat1[row*N + tx + k*tile_width],mat2[(ty + k*tile_width)*P + col]);
        __syncthreads();

        for (int l = 0; l < tile_width; l++) {
            double diff = tile1[ty*blockDim.y + l] - tile2[l*blockDim.y + tx];
            p_sum += diff*diff;
        }
        __syncthreads();
    }

    if (row < M && col < P) { 
        l2_sq[row*P + col] = p_sum;
    }
}

std::vector<int>
assign_centroids2(const std::vector<histogram_t>& h_descriptors, 
                  const std::vector<double>& h_centroids) {

    std::cout << "Lancer\n";
    std::cout << h_centroids.size() << "\n";
    int n_desc = h_descriptors.size();
    double *d_descriptors;
    cudaMalloc(&d_descriptors, n_desc * DESC_DIM * sizeof(double)); 
    cudaCheckError();

    int n_cent = h_centroids.size() / 256;
    double *d_centroids;
    cudaMalloc(&d_centroids, n_cent * DESC_DIM * sizeof(double)); 
    cudaCheckError();

    auto h_l2_squared = std::vector<double>(n_desc * n_cent);
    double *d_l2_squared; 
    cudaMalloc(&d_l2_squared, n_desc * n_cent * sizeof(double)); 
    cudaCheckError();

    auto h_assignments = std::vector<int>(n_desc);
    int *d_assignments;
    cudaMalloc(&d_assignments, n_desc * sizeof(int)); 
    cudaCheckError();

    cudaMemcpy(d_descriptors, &h_descriptors[0], n_desc * DESC_DIM * sizeof(double),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, &h_centroids[0], n_cent * DESC_DIM * sizeof(double),
               cudaMemcpyHostToDevice);
    cudaCheckError();

    // need to adjust because of shared memory usage -> SM
    //dim3 block_dim(32, 32);
    dim3 block_dim(32, 32);
    dim3 grid_dim((n_cent + block_dim.x - 1) / block_dim.x,
                  (n_desc + block_dim.y - 1) / block_dim.y);
    int patch_mem = block_dim.x*block_dim.y*sizeof(double);
    std::cout << grid_dim.x << "\n"
              << grid_dim.y << "\n"
              << patch_mem << "\n";

    l2_sq<<<grid_dim, block_dim, 2 * patch_mem>>>(d_descriptors, d_centroids, 
                                                  d_l2_squared, n_desc, DESC_DIM,
                                                  n_cent);
    cudaDeviceSynchronize();
    cudaCheckError();

    cudaMemcpy(&h_l2_squared[0], d_l2_squared, n_desc * n_cent * sizeof(double),
               cudaMemcpyDeviceToHost);
    cudaCheckError();

    //cudaMemcpy(&h_assignments[0], d_assignments, n_desc * sizeof(int),
    //           cudaMemcpyDeviceToHost);

    int i = 0;
    for (auto val : h_l2_squared){ 
        if (i % 32 == 0)
            std::cout << "\n";
        std::cout << val << " ";
        i++;
    }

    cudaFree(d_descriptors);
    cudaFree(d_centroids);
    cudaFree(d_l2_squared);
    cudaFree(d_assignments);
    cudaCheckError();

    return h_assignments;
}

}