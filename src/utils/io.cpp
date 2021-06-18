#include "io.hh"

#include <fstream>
#include <sstream>
#include <string>

namespace irgpu {

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

std::vector<double> load_descriptors_transpose(std::string filename) {

    std::ifstream input(filename);
    std::vector<double> descriptors_T;

    for(std::string line; std::getline(input, line); ) {
        std::stringstream in(line);

        double val;
        while (in >> val) {
            descriptors_T.push_back(val);
        }
    }

    return descriptors_T;
}

void save_pred(const std::vector<int>& pred, std::string filename) {
    std::ofstream output(filename);
    for (auto centroid : pred)
        output << centroid << "\n";
}

}