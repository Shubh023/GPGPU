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

std::vector<int>
assign_centroids(const std::vector<histogram_t>& h_descriptors, 
                 const std::vector<histogram_t>& h_centroids) {

    double *d_descriptors;
    double *d_centroids;
    int l_desc = h_descriptors.size() * 256;
    int l_cent = h_centroids.size() * 256;
    
    cudaMalloc(&d_descriptors, l_desc * sizeof(double)); 
    cudaMalloc(&d_centroids, l_cent * sizeof(double)); 
    cudaCheckError();

    cudaMemcpy(d_descriptors, &h_descriptors[0], l_desc * sizeof(double),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, &h_centroids[0], l_cent * sizeof(double),
               cudaMemcpyHostToDevice);
    cudaCheckError();

    cudaFree(d_descriptors);
    cudaFree(d_centroids);
    cudaCheckError();

    return std::vector<int>();
}

}