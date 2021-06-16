#include <fstream>
#include <sstream>
#include <limits>

#include "nn.hh"

namespace lbp {

std::vector<histogram_t> load_descriptors(std::string filename) {

    std::ifstream input(filename);
    std::vector<histogram_t> descriptors;

    for(std::string line; std::getline(input, line); ) {
        histogram_t desc{0};
        std::stringstream in(line);

        double val;
        int i = 0;
        while (in >> val) {
            desc[i] = val;
            i++;
        }

        descriptors.push_back(desc);
    }

    return descriptors;
}

double squared_L2_distance(const histogram_t& desc1, const histogram_t& desc2) {
    double res = 0;
    for (int i = 0; i < desc1.size(); i++) {
        double diff = desc1[i] - desc2[i];
        res += diff * diff;
    }
    return res;
}

int nearest_centroid(histogram_t desc, const std::vector<histogram_t>& centroids) {

    double min_dist = std::numeric_limits<double>::max();
    int best_centroid = -1;

    for (int i = 0; i < centroids.size(); i++) {
        float dist = squared_L2_distance(desc, centroids[i]);    
        if (dist < min_dist) {
            min_dist = dist;
            best_centroid = i;
        }
    }

    return best_centroid;
}

std::vector<int> assign_clusters(const std::vector<histogram_t>& descriptors, 
                                 const std::vector<histogram_t>& centroids) {
    std::vector<int> res;
    for (const auto& d : descriptors)
        res.push_back(nearest_centroid(d, centroids));

    return res;
}

void save_pred(const std::vector<int>& pred, std::string filename) {
    std::ofstream output(filename);
    for (auto centroid : pred)
        output << centroid << "\n";
}

}