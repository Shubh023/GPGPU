#include "io.hh"

#include <fstream>
#include <sstream>
#include <string>

namespace irgpu {

std::vector<histogram64_t> load_centroids(std::string filename) {

    std::ifstream input(filename);
    std::vector<histogram64_t> centroids;

    for(std::string line; std::getline(input, line); ) {
        histogram64_t desc{0};
        std::stringstream in(line);

        double val;
        int i = 0;
        while (in >> val) {
            desc[i] = val;
            i++;
        }

        centroids.push_back(desc);
    }

    return centroids;
}

std::vector<double> load_centroids_transpose(std::string filename) {

    std::ifstream input(filename);
    std::vector<double> centroids_T;

    for(std::string line; std::getline(input, line); ) {
        std::stringstream in(line);

        double val;
        while (in >> val) {
            centroids_T.push_back(val);
        }
    }

    return centroids_T;
}

void save_pred(const std::vector<int>& pred, std::string filename) {
    std::ofstream output(filename);
    for (auto centroid : pred)
        output << centroid << "\n";
}

}