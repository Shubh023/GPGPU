#include "lbp/lbp.hh"
#include "nearest-neighbor/nn_grid.hh"
#include "nearest-neighbor/nn_seq.hh"
#include "nearest-neighbor/nn_tiling.hh"
#include "pred.hh"
#include "utils/io.hh"

namespace irgpu {

std::vector<int> predict_centroids_seq(const cv::Mat& img) {
    std::vector<irgpu::histogram8_t> descriptors = irgpu::lbp_seq(img);
    auto centroids = irgpu::load_centroids("../resources/centroids.txt");
    auto pred = irgpu::assign_centroids_seq(descriptors, centroids);
    return pred;
}

std::vector<int> predict_centroids_gpu1(const cv::Mat& img) {
    std::vector<irgpu::histogram8_t> descriptors = irgpu::lbp_seq(img);
    auto centroids = irgpu::load_centroids("../resources/centroids.txt");
    auto pred = irgpu::assign_centroids_grid(descriptors, centroids);
    return pred;
}

}