#pragma once

#include <array>
#include <cstdint>
#include <opencv2/core/mat.hpp>

namespace irgpu {

    using histogram_t = std::array<double, 256>;

    //const int patch_size = 16;

    //std::array<uint8_t, 256> extract_textons(const cv::Mat& patch);
    //histogram_t hist(const std::array<uint8_t, 256>& textons);
    //std::vector<histogram_t> lbp(const cv::Mat& img);

}