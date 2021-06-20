#include <boost/timer/timer.hpp>
#include <iostream>
#include <opencv2/imgcodecs.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <vector>

#include <string>
#include "nearest-neighbor/nn_seq.hh"
#include "nearest-neighbor/nn_grid.hh"
#include "nearest-neighbor/nn_tiling.hh"
#include "utils/io.hh"

#define PH 16
#define PW 16

int main(int argc, char const *argv[]) {

    std::string image_path = "../../resources/beans.jpg";

    cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }

    std::vector<irgpu::histogram8_t> descriptors = irgpu::lbp_cuda(img);
    
    //auto centroids = irgpu::load_centroids("../resources/centroids.txt");
    //auto pred = irgpu::assign_centroids_seq(descriptors, centroids);
    //auto pred = irgpu::assign_centroids_grid(descriptors, centroids);

    auto centroids_T = irgpu::load_centroids_transpose("../../resources/centroids_t.txt");
    auto pred = irgpu::assign_centroids_tiling(descriptors, centroids_T);

    irgpu::save_pred(pred, "../resources/pred_cpp.txt");
    std::cout << pred.size() << std::endl;

    int index = 0;    
    cv::Mat reconstructed_image = img.clone();
    for (int r = 0; r < img.rows - PW; r += PW) {
        for (int c = 0; c < img.cols - PH; c += PH) {
            cv::Mat colorful = cv::Mat(PH, PW, CV_8UC1, pred[index]);
            cv::Mat dst_roi = reconstructed_image(cv::Rect(c, w, PH, PW));
            colorful.copyTo(dst_roi);
            index += 1;
        }
    }

    cv::Mat img_color;
    cv::applyColorMap(reconstructed_image, img_color, cv::COLORMAP_JET);
    cv::imshow("Reconstructed", img_color);
    cv::imshow("Original", img);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
