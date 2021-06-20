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

    int mode = 1;
    if (mode == 1)
    {
        std::string video_path = "../../resources/1_1080p60.MOV";
        cv::VideoCapture video_cap(video_path);
	    if(!video_cap.isOpened()){
	        std::cout << "Error opening video stream or file" << std::endl;
	        return -1;
	    }
	    while(1){
	        cv::Mat frame;
	        video_cap >> frame;
	        if (frame.empty())
	            break;
            ////////////////////

            
            cv::Mat resized = irgpu::resize_image(frame);
            cv::Mat grayscale;
            cv::cvtColor(resized, grayscale, cv::COLOR_BGR2GRAY);
            std::vector<irgpu::histogram8_t> descriptors = irgpu::lbp_cuda(grayscale);

            auto centroids_T = irgpu::load_centroids_transpose("../../resources/centroids_t.txt");
            auto pred = irgpu::assign_centroids_tiling(descriptors, centroids_T);

            irgpu::save_pred(pred, "../../resources/pred_cpp.txt");


            int index = 0;    
            cv::Mat reconstructed_image = grayscale.clone();
            for (int r = 0; r < grayscale.rows; r += PW) {
                for (int c = 0; c < grayscale.cols; c += PH) {
                    cv::Mat colorful = cv::Mat(PH, PW, CV_8UC1, pred.at(index));
                    cv::Mat dst_roi = reconstructed_image(cv::Rect(c, r, PH, PW));
                    colorful.copyTo(dst_roi);
                    index += 1;
                }
            }

            cv::Mat img_color;
            cv::normalize(reconstructed_image, img_color, 0, 255, cv::NORM_MINMAX);
            cv::applyColorMap(reconstructed_image, img_color, cv::COLORMAP_HSV);
            // cv::imshow("window", img_color);
            
            int scale = 0.8;
            int window_width = int(img_color.cols * scale);
            int window_height = int(img_color.rows * scale);
            cv::namedWindow("ROI Barcode", cv::WINDOW_NORMAL);
            cv::resizeWindow("ROI Barcode", window_height, window_width);
            cv::imshow("ROI Barcode", img_color);
            cv::namedWindow("Video", cv::WINDOW_NORMAL);
            cv::resizeWindow("Video", window_width, window_height);
            cv::imshow("Video", frame);

	        char c = (char)cv::waitKey(25);
	        if(c == 27)
	            break;
	    }
	    video_cap.release();
        cv::destroyAllWindows();
    }
    if (mode == 2) {

        std::string image_path = "../../resources/1.jpg";

        cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            std::cout << "Could not read the image: " << image_path << std::endl;
            return 1;
        }

        cv::Mat image = irgpu::resize_image(img);

        std::vector<irgpu::histogram8_t> descriptors = irgpu::lbp_cuda(image);
 
        auto centroids_T = irgpu::load_centroids_transpose("../../resources/centroids_t.txt");
        auto pred = irgpu::assign_centroids_tiling(descriptors, centroids_T);

        irgpu::save_pred(pred, "../../resources/pred_cpp.txt");

        int index = 0;    
        cv::Mat reconstructed_image = image.clone();
        for (int r = 0; r < image.rows; r += PW) {
            for (int c = 0; c < image.cols; c += PH) {
                // std::cout << pred.at(index) << " at " << index << std::endl;
                cv::Mat colorful = cv::Mat(PH, PW, CV_8UC1, pred.at(index));
                cv::Mat dst_roi = reconstructed_image(cv::Rect(c, r, PH, PW));
                colorful.copyTo(dst_roi);
                index += 1;
            }
        }
        
        cv::Mat img_color;
        cv::normalize(reconstructed_image, img_color, 0, 255, cv::NORM_MINMAX);
        cv::applyColorMap(reconstructed_image, img_color, cv::COLORMAP_HSV);
        cv::imshow("Reconstructed", img_color);
        cv::imshow("Original", image);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    
    return 0;
}
