#include <math.h>
#include "lbp.hh"
#include <bitset>

#define PH 16
#define PW 16

void check_cud(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
                  file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}
#define checkCudaErrors(val) check_cud( (val), #val, __FILE__, __LINE__ )

namespace irgpu {

cv::Mat resize_image(cv::Mat img)
{
    int width = round(img.cols , PW);
    int height = round(img.rows, PH);
    cv::Mat resized;
    // std::cout << "cols: " << img.cols << "\nrows: " << img.rows << std::endl;
    cv::resize(img, resized, cv::Size(width, height), 0, 0, cv::INTER_CUBIC);
    // std::cout << "\ncols: " << resized.cols << "\nrows: " << resized.rows << std::endl;
    return resized;
};

#define TILE_W   PW
#define TILE_H   PH
#define R        2
#define BLOCK_W  (TILE_W + R)
#define BLOCK_H  (TILE_H + R)

template <typename T>
__device__ inline T* eltPtr(T *baseAddress, int col, int row, size_t pitch) {
  return (T*)((char*)baseAddress + row * pitch + col * sizeof(int));
}

__device__ unsigned char pointLBP(unsigned char *in, const int row, const int col, const int w, const int h, int* d_histograms, size_t pitch)
{
    int index = row * w + col; 
    if (index > (w * h))
        return 0;
    
    unsigned char code = 0;
    unsigned char center = in[index];

    const int col_patch = col % PH;
    const int row_patch = row % PW;

    // UP LEFT
    if (row_patch != 0 || col_patch != 0)
    {
        code |= (in[index-w-1] >= center) << 7;
    }
    else
    {
        code |= (0 >= center) << 7;
    }
    // UP
    if (row_patch != 0)
    {            
        code |= (in[index-w  ] >= center) << 6;
    }
    else
    {            
        code |= (0 >= center) << 6;
    }
    // UP RIGHT
    if (row_patch != 0 || col_patch != (PW - 1))
    {
        code |= (in[index-w+1] >= center) << 5;
    }
    else
    {            
        code |= (0 >= center) << 5;
    }
    // LEFT
    if (col_patch != 0)
    {
        code |= (in[index  -1] >= center) << 4;
    }
    else
    {            
        code |= (0 >= center) << 4;
    }
    // RIGHT
    if (col_patch != (PW - 1))
    {
        code |= (in[index  +1] >= center) << 3;
    }
    else
    {            
        code |= (0 >= center) << 3;
    }
    // BOTTOM LEFT
    if (row_patch != (PW - 1) || col_patch != 0)
    {
        code |= (in[index+w-1] >= center) << 2;
    }
    else
    {            
        code |= (0 >= center) << 2;
    }
    // BOTTOM
    if (row_patch != (PW - 1))
    {
        code |= (in[index+w  ] >= center) << 1;
    }
    else
    {            
        code |= (0 >= center) << 1;
    }
    // BOTTOM RIGHT
    if (row_patch != (PW - 1) || col_patch != (PW - 1))
    {
        code |= (in[index+w+1] >= center) << 0;
    }
    else
    {            
        code |= (0 >= center) << 0;
    }
    
    // int trouver le patch sur le quel on est patch_index
    int patch_x = floor(float(row) / PW);
    int patch_y = floor(float(col) / PW);
    int patch_w = int(float(w)) / PW;
    int patch_index = patch_x * patch_w + patch_y;
    // d_histograms[patch_x * patch_w + patch_y][code] += 1;
    auto eptr = eltPtr<int>(d_histograms, int(code), patch_index, pitch);
    if (!eptr)
        *eptr = 0;
    atomicAdd(eptr, 1);
    // printf("\npind: %d, code: %d, eptr: %d", patch_index, int(code), *eptr);
    return code;
}

__global__ void LBP(unsigned char *in, const int w, const int h, int* d_histograms, size_t pitch) // , int* histograms (size num_patches * 256))
{
    //const unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    // int index = row + col * blockDim.y * gridDim.y;
    //TODO : Put histograms in shared memory for further optimizations...
    pointLBP(in, row, col, w, h, d_histograms, pitch);
}

void get_lbp(unsigned char *in, const int w, const int h, int* d_histograms, size_t pitch)
{
    const int
        sz = w * h * sizeof(unsigned char);
    unsigned char *in_gpu;
    checkCudaErrors(cudaMalloc((void**)&in_gpu,  sz));
    checkCudaErrors(cudaMemcpy(in_gpu,  in,  sz, cudaMemcpyHostToDevice));

    dim3 threadsperblock(PW, PH);
    int blockspergrid_x = int(ceil(float(w) / threadsperblock.x));
    int blockspergrid_y = int(ceil(float(h) / threadsperblock.y));
    dim3 blockspergrid(blockspergrid_x, blockspergrid_y);
    LBP<<<blockspergrid,threadsperblock>>>(in_gpu, w, h, d_histograms, pitch);
    checkCudaErrors(cudaThreadSynchronize());
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(in_gpu));
}

std::vector<histogram8_t> lbp_seq(const cv::Mat& image)
{
    int w = image.cols;
    int h = image.rows;
    // std::cout << w << " " << h << "\n";
   
    size_t num_patches = (floor(float(w)) / PW) *  (floor(float(h)) / PH);
    const int rows = num_patches;
    const int cols = 256;
    // int* host_buffer = new int[N_ARRAYS][256]
    int* host_buffer = (int*) std::malloc(rows * cols * sizeof(int));
    size_t pitch;
    int *d_buffer;
    // Allocate an 2D buffer with padding
    checkCudaErrors(cudaMallocPitch(&d_buffer, &pitch, cols * sizeof(int), rows));
    // printf("Pitch d_buffer: %ld\n", pitch);


    unsigned char* in = (unsigned char*) std::malloc(w * h * sizeof(unsigned char));
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            in[r * w + c] = u_char(image.at<uint8_t>(r,c));
        }
    }

    get_lbp(in, w, h, d_buffer, pitch);

    checkCudaErrors(cudaMemcpy2D(host_buffer, cols * sizeof(int), 
               d_buffer, pitch, cols * sizeof(int), rows, 
               cudaMemcpyDeviceToHost));


    std::vector<histogram8_t> histograms;
    for (int r = 0; r < rows; r++) {
        histogram8_t histogram{0};;
        for (int c = 0; c < cols; c++) {
            histogram[c] = host_buffer[r * cols + c];
        }
        histograms.push_back(histogram);
    }    
    checkCudaErrors(cudaFree(d_buffer));
    free(host_buffer);
    return histograms;
}

}