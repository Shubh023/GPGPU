//
// Created by shubh on 17/06/2021.
//

#pragma once
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <iostream>
#include <string>
#include <fstream>

namespace irgpu {
    inline int round(int x, int p) {
        return x - (x % p);
    }

    using histogram8_t = std::array<uint8_t, 256>;
    using histogram64_t = std::array<double, 256>;

    cv::Mat resize_image(cv::Mat img);

    std::vector<histogram8_t> lbp_seq(const cv::Mat& image);
}

