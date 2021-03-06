#include <iostream>
#include <stdio.h>

#include "nn_grid.hh"

#define DESC_DIM 256

//#define DEBUG     // comment to disable 

#define cudaCheckError() {                                                   \
    cudaError_t e=cudaGetLastError();                                        \
    if(e!=cudaSuccess) {                                                     \
        printf("Cuda failure %s:%d: '%s'\n", __FILE__ , __LINE__,            \
               cudaGetErrorString(e));                                       \
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
}


namespace irgpu {

__device__ double squared_L2_distance(uint8_t *desc, double *cent) {
    double res = 0;
    for (int i = 0; i < DESC_DIM; i++) {
        double diff = (double)desc[i] - cent[i];
        res += diff * diff;
    }
    return res;
}

__global__ void nearest_centroid(uint8_t *descriptors, double *centroids, 
                                 int *assignments, int n_desc, int n_cent) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n_desc)
        return; 

    double min_dist = squared_L2_distance(&descriptors[index * DESC_DIM],
                                          &centroids[0]);    
    int best_centroid = 0;

    for (int i = 1; i < n_cent ; i++) {
        double dist = squared_L2_distance(&descriptors[index * DESC_DIM],
                                          &centroids[i * DESC_DIM]);    
        if (dist < min_dist) {
            min_dist = dist;
            best_centroid = i;
        }
    }

    assignments[index] = best_centroid;
}

std::vector<int>
assign_centroids_grid(const std::vector<histogram8_t>& h_descriptors, 
                      const std::vector<histogram64_t>& h_centroids) {

    int n_desc = h_descriptors.size();
    uint8_t *d_descriptors;
    cudaMalloc(&d_descriptors, n_desc * DESC_DIM * sizeof(uint8_t)); 
    cudaCheckError();

    int n_cent = h_centroids.size();
    double *d_centroids;
    cudaMalloc(&d_centroids, n_cent * DESC_DIM * sizeof(double)); 
    cudaCheckError();

    auto h_assignments = std::vector<int>(n_desc);
    int *d_assignments;
    cudaMalloc(&d_assignments, n_desc * sizeof(int)); 
    cudaCheckError();

    cudaMemcpy(d_descriptors, &h_descriptors[0], n_desc * DESC_DIM * sizeof(uint8_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, &h_centroids[0], n_cent * DESC_DIM * sizeof(double),
               cudaMemcpyHostToDevice);
    cudaCheckError();

    int block_dim = 1024;
    int grid_dim = (n_desc + block_dim - 1) / block_dim;

#ifdef DEBUG
    std::cout << "Grid dim : " << grid_dim << "\n"
              << "Block dim : " << block_dim << "\n";
#endif

    nearest_centroid<<<grid_dim, block_dim>>>(d_descriptors, d_centroids,
                                              d_assignments, n_desc, n_cent);
    cudaDeviceSynchronize();
    cudaCheckError();

    cudaMemcpy(&h_assignments[0], d_assignments, n_desc * sizeof(int),
               cudaMemcpyDeviceToHost);

    cudaFree(d_descriptors);
    cudaFree(d_centroids);
    cudaFree(d_assignments);
    cudaCheckError();

    return h_assignments;
}

}