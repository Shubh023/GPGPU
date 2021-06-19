#include <limits>

#include "nn_seq.hh"
#include "lbp.hh"

namespace irgpu {

double squared_L2_distance(const histogram8_t& desc1, const histogram8_t& desc2) {
    double res = 0;
    for (int i = 0; i < desc1.size(); i++) {
        double diff = desc1[i] - desc2[i];
        res += diff * diff;
    }
    return res;
}

int nearest_centroid(histogram8_t desc, const std::vector<histogram8_t>& centroids) {

    double min_dist = std::numeric_limits<double>::max();
    int best_centroid = -1;

    for (int i = 0; i < centroids.size(); i++) {
        double dist = squared_L2_distance(desc, centroids[i]);    
        if (dist < min_dist) {
            min_dist = dist;
            best_centroid = i;
        }
    }

    return best_centroid;
}

std::vector<int> 
assign_centroids_seq(const std::vector<histogram8_t>& descriptors, 
                     const std::vector<histogram8_t>& centroids) {
    std::vector<int> res;
    for (const auto& d : descriptors)
        res.push_back(nearest_centroid(d, centroids));

    return res;
}

}