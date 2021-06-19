#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <string>

#include "nearest-neighbor/nn_seq.hh"
#include "nearest-neighbor/nn_grid.hh"
#include "nearest-neighbor/nn_tiling.hh"
#include "utils/io.hh"

#include <boost/timer/timer.hpp>

int main(int argc, char const *argv[]) {

    std::string image_path = "../resources/beans.jpg";

    cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if(img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }

    std::vector<irgpu::histogram_t>feature_vector = irgpu::lbp(img);
    
    //auto descriptors = irgpu::load_descriptors("../resources/desc.txt");
    //auto centroids = irgpu::load_descriptors("../resources/centroids.txt");
    //auto centroids_T = irgpu::load_descriptors_transpose("../resources/centroids_t.txt");

    //auto pred = irgpu::assign_centroids_seq(descriptors, centroids);
    //auto pred = irgpu::assign_centroids_grid(descriptors, centroids);
    //boost::timer::auto_cpu_timer t;
    //auto pred = irgpu::assign_centroids_tiling(descriptors, centroids_T);
    //irgpu::save_pred(pred, "../resources/pred_cpp.txt");
}