#pragma once

#include <vector>

#include "../lbp/lbp.hh"

namespace irgpu {

/**
 * @brief Assign a centroid to every descriptor in a list.
 *        Grid version : each thread is in charge of the assignment of one of
 *        descriptors.
 * 
 * @param descriptors Vector of descriptors.
 * @param centroids Vector of centroids.
 * @return std::vector<int> Vector of centroid indexes.
 */
std::vector<int>
assign_centroids_grid(const std::vector<histogram8_t>& descriptors, 
                      const std::vector<histogram64_t>& centroids);

}
