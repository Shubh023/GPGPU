#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <string>

#include "nn.hh"

int main(int argc, char const *argv[]) {
    
    auto descriptors = lbp::load_descriptors("../resources/desc.txt");
    auto centroids = lbp::load_descriptors("../resources/centroids.txt");

    auto pred = lbp::assign_clusters(descriptors, centroids);
    lbp::save_pred(pred, "../resources/pred_cpp.txt");
}