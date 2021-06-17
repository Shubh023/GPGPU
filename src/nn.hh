#pragma once

#include <array>
#include <vector>

namespace lbp {

using histogram_t = std::array<double, 256>;

/**
 * @brief Load descriptors from a text file to a vector. 
 * 
 * @param file File to load from.
 * @return std::vector<histogram_t> Vector of descriptors.
 */
std::vector<histogram_t> load_descriptors(std::string file);

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
std::vector<int> assign_clusters(const std::vector<histogram_t>& descriptors, 
                                 const std::vector<histogram_t>& centroids);

/**
 * @brief Save centroid indexes predictions to a file.
 * 
 * @param pred Vector of centroid indexes.
 * @param filename Output file.
 */
void save_pred(const std::vector<int>& pred, std::string filename);

}