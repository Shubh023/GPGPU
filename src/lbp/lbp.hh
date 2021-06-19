#pragma once

#include <vector>
#include <opencv2/core/mat.hpp>

namespace irgpu {

using histogram_t = std::array<double, 256>;


int round(int x, int p);

void display(cv::Mat mat);

std::vector<int> compared_neighbors(cv::Mat cell);

std::vector<std::vector<int>> extract_texton(cv::Mat patch);

cv::Mat padded(cv::Mat patch);

std::vector<std::vector<std::vector<int>>> 
textons_per_patch(std::vector<cv::Mat> patches);

int binary_to_int(std::vector<int> t);

histogram_t extract_hist(std::vector<std::vector<int>> texton);

std::vector<histogram_t>
get_histograms(std::vector<std::vector<std::vector<int>>> patches_textons);

std::string type2str(int type);

std::vector<cv::Mat> getPatches(cv::Mat img);

std::vector<histogram_t> lbp(cv::Mat img);

}

