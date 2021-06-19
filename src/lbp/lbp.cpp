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

#include "lbp.hh"

#define PH 16
#define PW 16

#ifdef DEBUG
#  define D(x) (x)
#else
#  define D(x) do{}while(0);
#endif


namespace irgpu {

int round(int x, int p) {
    return x - (x % p);
}

void display(cv::Mat mat) {
    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            std::cout << int(mat.at<uint8_t>(i,j)) << " ";
        }
        std::cout << std::endl;
    }
}

std::vector<int> compared_neighbors(cv::Mat cell) {
    std::vector<int> res;
    auto cell_center = int(cell.at<uint8_t>(1,1));
    for (int i = 0; i < cell.rows; ++i) {
        for (int j = 0; j < cell.cols; ++j) {
            if (i == 1 && j == 1) {
                continue;
            }
            if (int(cell.at<uint8_t>(i, j)) >= cell_center) {
                res.push_back(1);
            } else {
                res.push_back(0);
            }
        }
    }
    return res;
}

std::vector<std::vector<int>> extract_texton(cv::Mat patch) {
    std::vector<std::vector<int>> texton;
    for (int i = 0; i <= patch.rows - 3; ++i) {
        for (int j = 0; j <= patch.cols - 3; ++j) {
            cv::Mat cell = patch(cv::Rect(j, i, 3, 3));
            std::vector<int> compared_pixels = compared_neighbors(cell);
            texton.push_back(compared_pixels);
        }
    }
    return texton;
}

cv::Mat padded(cv::Mat patch) {
    cv::Mat padit;
    int padding = 1;
    padit.create(patch.rows + 2 * padding, patch.cols + 2 * padding, patch.type());
    padit.setTo(cv::Scalar::all(0));
    patch.copyTo(padit(cv::Rect(padding, padding, patch.cols, patch.rows)));
    return padit;
}

std::vector<std::vector<std::vector<int>>> 
textons_per_patch(std::vector<cv::Mat> patches) {

    std::vector<std::vector<std::vector<int>>> patches_textons;
    for (auto patch : patches)
        patches_textons.push_back(extract_texton(padded(patch)));
    return patches_textons;
}

int binary_to_int(std::vector<int> t) {
    int sum=0;
    for(int i=t.size()-1, j=0; i>=0; i--, j++){
        sum+=t[j]*(1<<i);
    }
    return sum;
}

histogram_t extract_hist(std::vector<std::vector<int>> texton) {
    histogram_t hist;
    int val = 0;
    for (auto t : texton) {
        val = binary_to_int(t);
        hist[255 - val] += 1;
    }
    return hist;
}

std::vector<histogram_t>
get_histograms(std::vector<std::vector<std::vector<int>>> patches_textons) {
    std::vector<histogram_t> histograms;
    for (auto pt : patches_textons)
        histograms.push_back(extract_hist(pt));
    return histograms;
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

std::vector<cv::Mat> getPatches(cv::Mat img) {
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

std::vector<histogram_t> lbp(cv::Mat img) {

    std::string ty =  type2str( img.type() );
    D(printf("Matrix: %s %dx%d \n", ty.c_str(), img.cols, img.rows);)

    // Resize image
    int width = round(img.cols, PW);
    int height = round(img.rows, PH);
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(width, height), 0, 0, cv::INTER_CUBIC);
    D(std::cout << "cols: " << img.cols << "\nrows: " << img.rows << std::endl;)
    D(std::cout << "\ncols: " << resized.cols << "\nrows: " << resized.rows << std::endl;)

    // Get all patches
    std::vector<cv::Mat> patches = getPatches(resized);
    D(std::cout << "Patches Extracted : " <<  patches.size() << std::endl;)

    // Compute 256 textons of each patches
    std::vector<std::vector<std::vector<int>>> patches_texton = textons_per_patch(patches);
    D(std::cout << "patches_texton shape : ("
                << patches_texton.size() << ","
                << int(patches_texton[0].size()) << ","
                << patches_texton[0][0].size() << ")" << std::endl;)

    // Compute textons histograms for each patch 
    std::vector<histogram_t> histograms = get_histograms(patches_texton);
    D(std::cout << "histograms shape : ("
                << histograms.size() << ","
                << int(histograms[0].size()) << ")" << std::endl;)

    D(
        auto h = histograms.at(0);
        for (int i = 0; i < 256; i++) {
            if (i % 11 == 0)
                std::cout << std::endl;
            std::cout << " " << h[i];
        }
        std::cout << std::endl;
        
        display(padded(patches.at(100)));
        std::vector<int> texton = patches_texton.at(100).at(0);
        for (auto e : texton)
            std::cout << " " << e;
        std::cout << std::endl;


        std::ofstream out("test.csv");

        for (auto& h : histograms) {
            for (auto b : h)
                out << b << ';';
            out << '\n';
        }
    )

    return histograms;
}

}

