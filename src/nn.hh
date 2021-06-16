#pragma once

#include <array>
#include <vector>

namespace lbp {

using histogram_t = std::array<double, 256>;

std::vector<histogram_t> load_descriptors(std::string file);

double squared_L2_distance(const histogram_t& desc1, const histogram_t& desc2);

int nearest_centroid(histogram_t desc, const std::vector<histogram_t>& centroids);

std::vector<int> assign_clusters(const std::vector<histogram_t>& descriptors, 
                                 const std::vector<histogram_t>& centroids);

}