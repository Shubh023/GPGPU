#pragma once

#include <array>
#include <cstdint>
#include <opencv2/core/mat.hpp>

namespace lbp {

    using histogram = std::array<uint8_t, 256>;

    const int patch_size = 16;

    uint8_t extract_texton(const cv::Mat& img, int x, int y);
    histogram hist(const std::array<uint8_t, 256>& textons);
    std::vector<histogram> lbp(const cv::Mat& img);

}