#include <vector>
#include <benchmark/benchmark.h>
#include <iostream>

#include "pred.hh"

constexpr int niteration = 1000;
const std::string image_path = "../../resources/1.jpg";


void BM_PredictCentroids_CPU(benchmark::State& st) {

    cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cout << "Could not read the image: " << image_path << std::endl;
    }

    for (auto _ : st)
        irgpu::predict_centroids_seq(img);

    st.counters["frame_rate"] = benchmark::Counter(st.iterations(),
                                                   benchmark::Counter::kIsRate);
}

void BM_PredictCentroids_GPU1(benchmark::State& st) {

    cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cout << "Could not read the image: " << image_path << std::endl;
    }

    for (auto _ : st)
        irgpu::predict_centroids_gpu1(img);

    st.counters["frame_rate"] = benchmark::Counter(st.iterations(),
                                                   benchmark::Counter::kIsRate);
}

void BM_PredictCentroids_GPU2(benchmark::State& st) {

    cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cout << "Could not read the image: " << image_path << std::endl;
    }

    for (auto _ : st)
        irgpu::predict_centroids_gpu2(img);

    st.counters["frame_rate"] = benchmark::Counter(st.iterations(),
                                                   benchmark::Counter::kIsRate);
}

BENCHMARK(BM_PredictCentroids_CPU)
->Unit(benchmark::kMillisecond)
->UseRealTime();

BENCHMARK(BM_PredictCentroids_GPU1)
->Unit(benchmark::kMillisecond)
->UseRealTime();

BENCHMARK(BM_PredictCentroids_GPU2)
->Unit(benchmark::kMillisecond)
->UseRealTime();

BENCHMARK_MAIN();