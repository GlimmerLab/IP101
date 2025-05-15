#include <basic/connected_components.hpp>
#include <queue>
#include <stack>
#include <omp.h>

namespace ip101 {

using namespace cv;
using namespace std;

namespace {
// Internal constant definitions
constexpr int CACHE_LINE = 64;    // CPU cache line size (bytes)
constexpr int BLOCK_SIZE = 16;    // Block processing size

// Disjoint Set implementation
class DisjointSet {
public:
    DisjointSet(int size) : parent(size), rank(size, 0) {
        for (int i = 0; i < size; i++) {
            parent[i] = i;
        }
    }

    int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]); // Path compression
        }
        return parent[x];
    }

    void unite(int x, int y) {
        x = find(x);
        y = find(y);
        if (x == y) return;

        // Union by rank
        if (rank[x] < rank[y]) {
            parent[x] = y;
        } else {
            parent[y] = x;
            if (rank[x] == rank[y]) {
                rank[x]++;
            }
        }
    }

private:
    vector<int> parent;
    vector<int> rank;
};

// Two-pass algorithm for 4-connected labeling
int two_pass_4connected(const Mat& src, Mat& labels) {
    int height = src.rows;
    int width = src.cols;

    // First pass: initial labeling
    labels = Mat::zeros(height, width, CV_32S);
    int current_label = 1;
    DisjointSet ds(height * width / 4); // Estimated label count

    #pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (src.at<uchar>(y, x) == 0) continue;

            vector<int> neighbor_labels;
            // Check pixels above and to the left
            if (y > 0 && labels.at<int>(y-1, x) > 0)
                neighbor_labels.push_back(labels.at<int>(y-1, x));
            if (x > 0 && labels.at<int>(y, x-1) > 0)
                neighbor_labels.push_back(labels.at<int>(y, x-1));

            if (neighbor_labels.empty()) {
                // New component
                labels.at<int>(y, x) = current_label++;
            } else {
                // Take minimum label
                int min_label = *min_element(neighbor_labels.begin(), neighbor_labels.end());
                labels.at<int>(y, x) = min_label;
                // Merge equivalent labels
                for (int label : neighbor_labels) {
                    ds.unite(min_label-1, label-1);
                }
            }
        }
    }

    // Second pass: resolve label equivalences
    vector<int> label_map(current_label);
    int num_labels = 0;
    for (int i = 0; i < current_label; i++) {
        if (ds.find(i) == i) {
            label_map[i] = ++num_labels;
        }
    }
    for (int i = 0; i < current_label; i++) {
        label_map[i] = label_map[ds.find(i)];
    }

    #pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (labels.at<int>(y, x) > 0) {
                labels.at<int>(y, x) = label_map[labels.at<int>(y, x)-1];
            }
        }
    }

    return num_labels;
}

// Two-pass algorithm for 8-connected labeling
int two_pass_8connected(const Mat& src, Mat& labels) {
    int height = src.rows;
    int width = src.cols;

    // First pass: initial labeling
    labels = Mat::zeros(height, width, CV_32S);
    int current_label = 1;
    DisjointSet ds(height * width / 4); // Estimated label count

    #pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (src.at<uchar>(y, x) == 0) continue;

            vector<int> neighbor_labels;
            // Check 8-neighborhood pixels
            for (int dy = -1; dy <= 0; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (dy == 0 && dx >= 0) break;
                    int ny = y + dy;
                    int nx = x + dx;
                    if (ny >= 0 && nx >= 0 && nx < width) {
                        if (labels.at<int>(ny, nx) > 0) {
                            neighbor_labels.push_back(labels.at<int>(ny, nx));
                        }
                    }
                }
            }

            if (neighbor_labels.empty()) {
                // New component
                labels.at<int>(y, x) = current_label++;
            } else {
                // Take minimum label
                int min_label = *min_element(neighbor_labels.begin(), neighbor_labels.end());
                labels.at<int>(y, x) = min_label;
                // Merge equivalent labels
                for (int label : neighbor_labels) {
                    ds.unite(min_label-1, label-1);
                }
            }
        }
    }

    // Second pass: resolve label equivalences
    vector<int> label_map(current_label);
    int num_labels = 0;
    for (int i = 0; i < current_label; i++) {
        if (ds.find(i) == i) {
            label_map[i] = ++num_labels;
        }
    }
    for (int i = 0; i < current_label; i++) {
        label_map[i] = label_map[ds.find(i)];
    }

    #pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (labels.at<int>(y, x) > 0) {
                labels.at<int>(y, x) = label_map[labels.at<int>(y, x)-1];
            }
        }
    }

    return num_labels;
}

} // anonymous namespace

int label_4connected(const Mat& src, Mat& labels) {
    CV_Assert(!src.empty() && src.type() == CV_8UC1);
    return two_pass_4connected(src, labels);
}

int label_8connected(const Mat& src, Mat& labels) {
    CV_Assert(!src.empty() && src.type() == CV_8UC1);
    return two_pass_8connected(src, labels);
}

vector<ConnectedComponent> analyze_components(const Mat& labels, int num_labels) {
    vector<ConnectedComponent> stats(num_labels);

    // Initialize statistics
    for (int i = 0; i < num_labels; i++) {
        stats[i].label = i + 1;
        stats[i].area = 0;
        stats[i].bbox = Rect(labels.cols, labels.rows, 0, 0);
        stats[i].centroid = Point(0, 0);
    }

    // Calculate basic properties
    #pragma omp parallel for
    for (int y = 0; y < labels.rows; y++) {
        for (int x = 0; x < labels.cols; x++) {
            int label = labels.at<int>(y, x);
            if (label == 0) continue;

            ConnectedComponent& comp = stats[label-1];
            #pragma omp atomic
            comp.area++;

            #pragma omp critical
            {
                comp.bbox.x = min(comp.bbox.x, x);
                comp.bbox.y = min(comp.bbox.y, y);
                comp.bbox.width = max(comp.bbox.width, x - comp.bbox.x + 1);
                comp.bbox.height = max(comp.bbox.height, y - comp.bbox.y + 1);
                comp.centroid.x += x;
                comp.centroid.y += y;
            }
        }
    }

    // Calculate advanced properties
    for (auto& comp : stats) {
        if (comp.area > 0) {
            comp.centroid.x /= comp.area;
            comp.centroid.y /= comp.area;

            // Calculate circularity
            double perimeter = 0;
            for (int y = comp.bbox.y; y < comp.bbox.y + comp.bbox.height; y++) {
                for (int x = comp.bbox.x; x < comp.bbox.x + comp.bbox.width; x++) {
                    if (labels.at<int>(y, x) == comp.label) {
                        // Check boundary point
                        bool is_boundary = false;
                        for (int dy = -1; dy <= 1; dy++) {
                            for (int dx = -1; dx <= 1; dx++) {
                                int ny = y + dy;
                                int nx = x + dx;
                                if (ny >= 0 && ny < labels.rows && nx >= 0 && nx < labels.cols) {
                                    if (labels.at<int>(ny, nx) != comp.label) {
                                        is_boundary = true;
                                        break;
                                    }
                                }
                            }
                            if (is_boundary) break;
                        }
                        if (is_boundary) perimeter++;
                    }
                }
            }
            comp.circularity = 4 * CV_PI * comp.area / (perimeter * perimeter);

            // Calculate aspect ratio
            comp.aspect_ratio = (double)comp.bbox.width / comp.bbox.height;

            // Calculate solidity
            comp.solidity = (double)comp.area / (comp.bbox.width * comp.bbox.height);
        }
    }

    return stats;
}

Mat filter_components(const Mat& labels,
                     const vector<ConnectedComponent>& stats,
                     int min_area,
                     int max_area) {
    Mat filtered = Mat::zeros(labels.size(), labels.type());

    #pragma omp parallel for
    for (int y = 0; y < labels.rows; y++) {
        for (int x = 0; x < labels.cols; x++) {
            int label = labels.at<int>(y, x);
            if (label == 0) continue;

            const auto& comp = stats[label-1];
            if (comp.area >= min_area && comp.area <= max_area) {
                filtered.at<int>(y, x) = label;
            }
        }
    }

    return filtered;
}

Mat draw_components(const Mat& src,
                   const Mat& labels,
                   const vector<ConnectedComponent>& stats) {
    Mat colored;
    cvtColor(src, colored, COLOR_GRAY2BGR);

    // Generate random colors
    RNG rng(12345);
    vector<Vec3b> colors(stats.size());
    for (size_t i = 0; i < stats.size(); i++) {
        colors[i] = Vec3b(rng.uniform(0, 256),
                         rng.uniform(0, 256),
                         rng.uniform(0, 256));
    }

    // Draw connected components
    #pragma omp parallel for
    for (int y = 0; y < labels.rows; y++) {
        for (int x = 0; x < labels.cols; x++) {
            int label = labels.at<int>(y, x);
            if (label > 0) {
                colored.at<Vec3b>(y, x) = colors[label-1];
            }
        }
    }

    // Draw bounding boxes and properties
    for (const auto& comp : stats) {
        if (comp.area > 0) {
            rectangle(colored, comp.bbox, Scalar(0, 255, 0), 2);
            circle(colored, comp.centroid, 3, Scalar(0, 0, 255), -1);

            string info = format("Label: %d, Area: %d", comp.label, comp.area);
            putText(colored, info, Point(comp.bbox.x, comp.bbox.y-5),
                   FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
        }
    }

    return colored;
}

} // namespace ip101