#pragma once

#include <vector>

#include "../lbp/lbp.hh"

#define DESC_DIM 256


namespace irgpu {

/**
 * @brief Assign a centroid to every descriptor in a list.
 * 
 * @param descriptors Vector of descriptors.
 * @param centroids Vector of centroids.
 * @return std::vector<int> Vector of centroid indexes.
 */
std::vector<int>
assign_centroids2(const std::vector<histogram_t>& descriptors, 
                 const std::vector<double>& centroids_T);

}
