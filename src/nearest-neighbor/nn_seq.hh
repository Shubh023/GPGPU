#pragma once

#include <vector>

#include "../lbp/lbp.hh"


namespace irgpu {

/**
 * @brief Squared euclidean distance between two descriptors.
 *        Sequential version.
 * 
 * @param desc1 First descriptor.
 * @param desc2 Second descriptor.
 * @return double Squared L2 distance.
 */
static double squared_L2_distance(const histogram8_t& desc1,
                                  const histogram8_t& desc2);

/**
 * @brief Find the index of the nearest centroid of a descriptor.
 * 
 * @param desc Descriptor to assign a centroid to.
 * @param centroids Vector of centroids. 
 * @return int Index of the centroid assigned.
 */
static int nearest_centroid(histogram8_t desc,
                            const std::vector<histogram8_t>& centroids);

/**
 * @brief Assign a centroid to every descriptor in a list.
 * 
 * @param descriptors Vector of descriptors.
 * @param centroids Vector of centroids.
 * @return std::vector<int> Vector of centroid indexes.
 */
std::vector<int>
assign_centroids_seq(const std::vector<histogram8_t>& descriptors, 
                     const std::vector<histogram8_t>& centroids);

}