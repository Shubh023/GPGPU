#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <string>

#include "nearest-neighbor/nn_seq.hh"
#include "nearest-neighbor/nn.hh"
#include "nearest-neighbor/nn2.hh"
#include "utils/io.hh"

int main(int argc, char const *argv[]) {
    
    auto descriptors = irgpu::load_descriptors("../resources/desc.txt");
    //auto centroids = irgpu::load_descriptors("../resources/centroids.txt");
    auto centroids_T = irgpu::load_descriptors_transpose("../resources/centroids_t.txt");

    //auto pred = irgpu::assign_centroids_seq(descriptors, centroids);
    //auto pred = irgpu::assign_centroids(descriptors, centroids);
    auto pred = irgpu::assign_centroids2(descriptors, centroids_T);
    irgpu::save_pred(pred, "../resources/pred_cpp.txt");
}