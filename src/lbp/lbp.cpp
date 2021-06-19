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
#include <assert.h>

#include "lbp.hh"

#define PH 16
#define PW 16

#ifndef DEBUG 
#  define DEBUG 1 // set debug mode
#endif


namespace irgpu {

int round(int x, int p) {
    return x - (x % p);
}

void display(const cv::Mat& mat) {
    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            std::cout << int(mat.at<uint8_t>(i,j)) << " ";
        }
        std::cout << std::endl;
    }
}

std::string type2str(int type) {
    std::string r;

    uint8_t depth = type & CV_MAT_DEPTH_MASK;
    uint8_t chans = 1 + (type >> CV_CN_SHIFT);

    switch ( depth ) {
        case CV_8U:  r = "8U"; break;
        case CV_8S:  r = "8S"; break;
        case CV_16U: r = "16U"; break;
        case CV_16S: r = "16S"; break;
        case CV_32S: r = "32S"; break;
        case CV_32F: r = "32F"; break;
        case CV_64F: r = "64F"; break;
        default:     r = "User"; break;
    }

    r += "C";
    r += (chans+'0');

    return r;
}

std::vector<cv::Mat> get_patches(const cv::Mat& img) {
    std::vector<cv::Mat> patches;
    cv::Mat patch;

    for (int j = 0; j < img.rows - PH; j += PH) {
        for (int i = 0; i < img.cols - PW; i += PW) {
            patch = img(cv::Rect(i, j, PH, PW));
            patches.push_back(patch);
        }
    }
    return patches;
}

std::vector<uint8_t> extract_textons(const cv::Mat& patch) {
    std::vector<uint8_t> textons;

    uint8_t m, c;

    for (unsigned x = 1; x < (patch.cols - 1); x++) {
        for (unsigned y = 1; y < (patch.rows - 1); y++) {
            m = 0;
            c = patch.at<uint8_t>(x, y);

            //std::cout << +patch.at<uint8_t>(x - 1, y - 1) << " ";
            //std::cout << +patch.at<uint8_t>(x - 1, y    ) << " ";
            //std::cout << +patch.at<uint8_t>(x - 1, y + 1) << " ";
            //std::cout << +patch.at<uint8_t>(x,     y - 1) << " ";
            //std::cout << +patch.at<uint8_t>(x,     y    ) << " ";
            //std::cout << +patch.at<uint8_t>(x,     y + 1) << " ";
            //std::cout << +patch.at<uint8_t>(x + 1, y - 1) << " ";
            //std::cout << +patch.at<uint8_t>(x + 1, y    ) << " ";
            //std::cout << +patch.at<uint8_t>(x + 1, y + 1) << " ";

            m |= (patch.at<uint8_t>(x - 1, y - 1) > c) << 7; // (-1, -1)
            m |= (patch.at<uint8_t>(x - 1, y    ) > c) << 6; // (-1,  0)
            m |= (patch.at<uint8_t>(x - 1, y + 1) > c) << 5; // (-1,  1)
            m |= (patch.at<uint8_t>(x,     y - 1) > c) << 4; // ( 0, -1)
            m |= (patch.at<uint8_t>(x,     y + 1) > c) << 3; // ( 0,  1)
            m |= (patch.at<uint8_t>(x + 1, y - 1) > c) << 2; // ( 1, -1)
            m |= (patch.at<uint8_t>(x + 1, y    ) > c) << 1; // ( 1,  0)
            m |= (patch.at<uint8_t>(x + 1, y + 1) > c) << 0; // ( 1,  1)

            textons.push_back(m);
        }
    }

    return textons;
}

cv::Mat padded(const cv::Mat& patch) {
    cv::Mat padit;
    int padding = 1;
    padit.create(patch.rows + 2 * padding, patch.cols + 2 * padding, patch.type());
    padit.setTo(cv::Scalar::all(0));
    patch.copyTo(padit(cv::Rect(padding, padding, patch.cols, patch.rows)));
    return padit;
}

std::vector<std::vector<uint8_t>>
textons_per_patch(const std::vector<cv::Mat>& patches) {
    std::vector<std::vector<uint8_t>> patches_textons;
    for (const auto& patch : patches) {
        patches_textons.push_back(extract_textons(padded(patch)));
    }
    return patches_textons;
}

histogram8_t extract_hist(const std::vector<uint8_t>& textons) {
    histogram8_t hist{0};
    for (uint8_t t : textons) {
        hist[t] += 1;
    }
    return hist;
}

std::vector<histogram8_t>
get_histograms(const std::vector<std::vector<uint8_t>>& patches_textons) {
    std::vector<histogram8_t> histograms;
    for (const auto& textons : patches_textons)
        histograms.push_back(extract_hist(textons));
    return histograms;
}

std::vector<histogram8_t> lbp_seq(const cv::Mat& img) {

    std::string ty =  type2str(img.type());
    printf("Matrix: %s %dx%d \n", ty.c_str(), img.cols, img.rows);

    // Resize image
    int width = round(img.cols, PW);
    int height = round(img.rows, PH);
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(width, height), 0, 0, cv::INTER_CUBIC);
    std::cout << "cols: " << img.cols << "\nrows: " << img.rows << std::endl;
    std::cout << "\ncols: " << resized.cols << "\nrows: " << resized.rows << std::endl;

    // Get all patches
    std::vector<cv::Mat> patches = get_patches(resized);
    std::cout << "Patches Extracted : " <<  patches.size() << std::endl;

    // Compute 256 textons of each patches
    std::vector<std::vector<uint8_t>> patches_texton = textons_per_patch(patches);
    std::cout << "patches_texton shape : ("
                << patches_texton.size() << ","
                << patches_texton[0].size() << ")" << std::endl;

    // Compute textons histograms for each patch 
    std::vector<histogram8_t> histograms = get_histograms(patches_texton);
    std::cout << "histograms shape : ("
                << histograms.size() << ","
                << histograms[0].size() << ")" << std::endl;

#ifdef DEBUG
    for (const auto& h : histograms) {
        uint8_t sum = 0;
        for (uint8_t val : h) {
            sum += val;
        }
        if (sum != 0) {
            std::cerr << "Missing values in histogram.\n";
        }
    }
#endif

#ifdef DEBUG

    //std::cout << "Patch 100 :\n";
    //display(padded(patches.at(100)));

    //std::cout << "Texton 0 of patch 100 :\n";
    //std::vector<int> texton = patches_texton.at(100).at(0);
    //for (auto e : texton)
    //    std::cout << " " << e;
    //std::cout << std::endl;

    //std::cout << "Histograms :\n";
    //std::ofstream out("test.csv");
    //for (auto& h : histograms) {
    //    for (auto b : h)
    //        out << b << ';';
    //    out << '\n';
    //}

    //for (auto b : histograms[0])
    //    std::cout << b << ';';

#endif

    return histograms;
}

}

