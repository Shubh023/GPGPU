#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "lbp.hh"

namespace lbp {


uint8_t extract_texton(const cv::Mat& img, int x, int y) {
    return 0;
}

histogram hist(const std::array<uint8_t, 256>& textons) {
    return std::array<uint8_t, 256>{ 0 };
}

std::pair<int, int> padding_size(int n, int patch_size) {

    int diff = patch_size - n % patch_size;
    std::pair<int, int> pad(diff / 2, diff / 2);
    if (diff % 2 != 0) {
        pad.second++;
    }

    return pad;
}

std::vector<histogram> lbp(const cv::Mat& img) {

    int rows = img.rows;
    int cols = img.cols;

    std::pair<int, int> x_pad = padding_size(rows, patch_size);
    std::pair<int, int> y_pad = padding_size(rows, patch_size);

    cv::Mat padded_img = cv::Mat(rows + x_pad.first + x_pad.second,
                                 cols + y_pad.first + y_pad.second,
                                 img.depth());
    
    cv::copyMakeBorder(img, padded_img, x_pad.first, x_pad.second,
                       y_pad.first, y_pad.second, cv::BORDER_REPLICATE);

    int n_x = padded_img.rows / patch_size;
    int n_y = padded_img.cols / patch_size;

    for (int i = 0; i < n_x; i++) {
        for (int j = 0; j < n_y; j++) {
            cv::Rect patch(j * patch_size, i * patch_size, patch_size, patch_size);
            Mat patch = padded_img(patch));
        }
    }

    return std::vector<histogram>();
}

}