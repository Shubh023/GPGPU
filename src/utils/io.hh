#pragma once 

#include <vector>

#include "lbp.hh"

namespace irgpu {

/**
 * @brief Load descriptors from a text file to a vector. 
 * 
 * @param file File to load from.
 * @return std::vector<histogram_t> Vector of descriptors.
 */
std::vector<histogram_t> load_descriptors(std::string file);

/**
 * @brief Load tranpose of descriptors from a text file to a vector. 
 *        Used for centroids.
 * 
 * @param file File to load from.
 * @return std::vector<double> Linearized transpose matrix.
 */
std::vector<double> load_descriptors_transpose(std::string file);

/**
 * @brief Save centroid indexes predictions to a file.
 * 
 * @param pred Vector of centroid indexes.
 * @param filename Output file.
 */
void save_pred(const std::vector<int>& pred, std::string filename);

}