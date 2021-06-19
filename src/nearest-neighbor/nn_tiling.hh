#pragma once

#include <vector>

#include "../lbp/lbp.hh"


namespace irgpu {

/**
 * @brief Assign a centroid to every descriptor in a list.
 *        Version using the tiling method.
 * 
 * @param descriptors Vector of descriptors.
 * @param centroids Vector of centroids.
 * @return std::vector<int> Vector of centroid indexes.
 */
std::vector<int>
assign_centroids_tiling(const std::vector<histogram8_t>& descriptors, 
                        const std::vector<double>& centroids_T);

}
