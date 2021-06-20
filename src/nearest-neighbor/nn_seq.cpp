#include <limits>

#include "nn_seq.hh"
#include "lbp.hh"

//#define DEBUG     // comment to disable 


namespace irgpu {

double squared_L2_distance(const histogram8_t& desc, const histogram64_t& cent) {

    double res = 0;
    for (int i = 0; i < desc.size(); i++) {
        double diff = (double)desc[i] - cent[i];
        res += diff * diff;
    }
    return res;
}

int nearest_centroid(const histogram8_t& desc,
                     const std::vector<histogram64_t>& centroids) {

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
                     const std::vector<histogram64_t>& centroids) {

    std::vector<int> res;
    for (const auto& d : descriptors)
        res.push_back(nearest_centroid(d, centroids));

    return res;
}

}