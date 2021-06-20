#pragma once

#include <opencv2/imgcodecs.hpp>
#include <vector>

namespace irgpu {

std::vector<int> predict_centroids_seq(const cv::Mat& img);

std::vector<int> predict_centroids_gpu1(const cv::Mat& img);

}