#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <string>

#include "nearest-neighbor/nn_seq.hh"
#include "utils/io.hh"

int main(int argc, char const *argv[]) {
    
    auto descriptors = irgpu::load_descriptors("../resources/desc.txt");
    auto centroids = irgpu::load_descriptors("../resources/centroids.txt");

    auto pred = irgpu::assign_centroids_seq(descriptors, centroids);
    irgpu::save_pred(pred, "../resources/pred_cpp.txt");
}