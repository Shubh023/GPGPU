#include <iostream>
#include <stdio.h>

#include "nn.hh"

#define cudaCheckError() {                                                   \
    cudaError_t e=cudaGetLastError();                                        \
    if(e!=cudaSuccess) {                                                     \
        printf("Cuda failure %s:%d: '%s'\n", __FILE__ , __LINE__,            \
               cudaGetErrorString(e));                                       \
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
}

namespace irgpu {

__device__ double squared_L2_distance(double* desc1, double* desc2) {
    double res = 0;
    for (int i = 0; i < DESC_SIZE; i++) {
        double diff = desc1[i] - desc2[i];
        res += diff * diff;
    }
    return res;
}

__global__ void nearest_centroid(double *descriptors, double *centroids, 
                                 int *assignments, int n_desc, int n_cent) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n_desc)
        return; 

    double min_dist = squared_L2_distance(&descriptors[index * DESC_SIZE],
                                          &centroids[0]);    
    int best_centroid = 0;
    //printf("%f %d\n", min_dist, best_centroid);

    for (int i = 1; i < n_cent ; i++) {
        double dist = squared_L2_distance(&descriptors[index * DESC_SIZE],
                                          &centroids[i * DESC_SIZE]);    
        if (dist < min_dist) {
            min_dist = dist;
            best_centroid = i;
            //printf("%f %d\n", min_dist, i);
        }
    }

    assignments[index] = best_centroid;
}

std::vector<int>
assign_centroids(const std::vector<histogram_t>& h_descriptors, 
                 const std::vector<histogram_t>& h_centroids) {

    std::cout << "Lancer\n";
    int n_desc = h_descriptors.size();
    double *d_descriptors;
    cudaMalloc(&d_descriptors, n_desc * DESC_SIZE * sizeof(double)); 
    cudaCheckError();

    int n_cent = h_centroids.size();
    double *d_centroids;
    cudaMalloc(&d_centroids, n_cent * DESC_SIZE* sizeof(double)); 
    cudaCheckError();

    auto h_assignments = std::vector<int>(n_desc);
    int *d_assignments;
    cudaMalloc(&d_assignments, n_desc * sizeof(int)); 
    cudaCheckError();

    cudaMemcpy(d_descriptors, &h_descriptors[0], n_desc * DESC_SIZE * sizeof(double),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, &h_centroids[0], n_cent * DESC_SIZE * sizeof(double),
               cudaMemcpyHostToDevice);
    cudaCheckError();

    int threads_per_block = 1024;
    int blocks_per_grid = (n_desc + threads_per_block - 1) / threads_per_block;
    std::cout << blocks_per_grid << " " << threads_per_block << "\n";
    nearest_centroid<<<blocks_per_grid, threads_per_block>>>(d_descriptors,
                                                             d_centroids,
                                                             d_assignments,
                                                             n_desc, n_cent);
    cudaDeviceSynchronize();
    cudaCheckError();

    cudaMemcpy(&h_assignments[0], d_assignments, n_desc * sizeof(int),
               cudaMemcpyDeviceToHost);

    //for (auto val : h_assignments)
    //    std::cout << val << " ";

    cudaFree(d_descriptors);
    cudaFree(d_centroids);
    cudaFree(d_assignments);
    cudaCheckError();

    return h_assignments;
}

}