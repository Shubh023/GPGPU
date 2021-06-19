#pragma once

#include <vector>
#include <opencv2/core/mat.hpp>

namespace irgpu {

using histogram_t = std::array<uint8_t, 256>;


int round(int x, int p);

void display(const cv::Mat& mat);

std::string type2str(int type);

std::vector<cv::Mat> get_patches(const cv::Mat& img);

std::vector<int> compared_neighbors(const cv::Mat& cell);

std::vector<std::vector<int>> extract_texton(const cv::Mat& patch);

std::vector<uint8_t> extract_textons(const cv::Mat& patch);

cv::Mat padded(const cv::Mat& patch);

std::vector<std::vector<std::vector<int>>> 
textons_per_patch(const std::vector<cv::Mat>& patches);

int binary_to_int(const std::vector<int>& t);

histogram_t extract_hist(const std::vector<std::vector<int>>& texton);

std::vector<histogram_t>
get_histograms(const std::vector<std::vector<std::vector<int>>>& patches_textons);

std::vector<histogram_t> lbp_seq(const cv::Mat& img);

}

