#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <string>

#include "lbp.hh"

int main(int argc, char const *argv[]) {
    
    std::string image_path = "../resources/beans.jpg";

    cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if(img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }

    std::vector<lbp::histogram>feature_vector = lbp::lbp(img);
}