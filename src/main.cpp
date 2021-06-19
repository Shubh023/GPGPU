#include <boost/timer/timer.hpp>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <string>
#include <benchmark/benchmark.h>

#include "nearest-neighbor/nn_seq.hh"
#include "nearest-neighbor/nn_grid.hh"
#include "nearest-neighbor/nn_tiling.hh"
#include "utils/io.hh"


int main(int argc, char const *argv[]) {

    std::string image_path = "../resources/beans.jpg";

    cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }

    std::vector<irgpu::histogram8_t> descriptors = irgpu::lbp_seq(img);
    
    //auto centroids = irgpu::load_centroids("../resources/centroids.txt");
    //auto pred = irgpu::assign_centroids_seq(descriptors, centroids);
    //auto pred = irgpu::assign_centroids_grid(descriptors, centroids);

    auto centroids_T = irgpu::load_centroids_transpose("../resources/centroids_t.txt");
    auto pred = irgpu::assign_centroids_tiling(descriptors, centroids_T);

    irgpu::save_pred(pred, "../resources/pred_cpp.txt");
}