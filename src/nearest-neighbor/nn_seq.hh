#pragma once

#include <array>
#include <vector>

#include "lbp.hh"


namespace irgpu {

/**
 * @brief Squared euclidean distance between two descriptors.
 * 
 * @param desc1 First descriptor.
 * @param desc2 Second descriptor.
 * @return double Squared L2 distance.
 */
double squared_L2_distance(const histogram_t& desc1, const histogram_t& desc2);

/**
 * @brief Find the index of the nearest centroid of a descriptor.
 * 
 * @param desc Descriptor to assign a centroid to.
 * @param centroids Vector of centroids. 
 * @return int Index of the centroid assigned.
 */
int nearest_centroid(histogram_t desc, const std::vector<histogram_t>& centroids);

/**
 * @brief Assign a centroid to every descriptor in a list.
 * 
 * @param descriptors Vector of descriptors.
 * @param centroids Vector of centroids.
 * @return std::vector<int> Vector of centroid indexes.
 */
std::vector<int>
assign_centroids_seq(const std::vector<histogram_t>& descriptors, 
                     const std::vector<histogram_t>& centroids);

}