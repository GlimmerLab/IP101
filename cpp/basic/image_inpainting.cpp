#include <basic/image_inpainting.hpp>
#include <omp.h>
#include <queue>
#include <cmath>

namespace ip101 {

using namespace cv;
using namespace std;

// Custom comparator for Point class
struct PointCompare {
    bool operator()(const Point& a, const Point& b) const {
        return (a.y < b.y) || (a.y == b.y && a.x < b.x);
    }
};

namespace {
// Internal constants
constexpr int CACHE_LINE = 64;    // CPU cache line size (bytes)
constexpr int BLOCK_SIZE = 16;    // Block processing size
constexpr double EPSILON = 1e-6;   // Numerical precision

// Calculate patch similarity
double compute_patch_similarity(
    const Mat& src1,
    const Mat& src2,
    const Point& p1,
    const Point& p2,
    int patch_size) {

    double sum = 0;
    int half = patch_size / 2;

    #pragma omp parallel for reduction(+:sum)
    for(int i = -half; i <= half; i++) {
        for(int j = -half; j <= half; j++) {
            Point pt1 = p1 + Point(j, i);
            Point pt2 = p2 + Point(j, i);

            if(pt1.x >= 0 && pt1.x < src1.cols &&
               pt1.y >= 0 && pt1.y < src1.rows &&
               pt2.x >= 0 && pt2.x < src2.cols &&
               pt2.y >= 0 && pt2.y < src2.rows) {
                Vec3b v1 = src1.at<Vec3b>(pt1);
                Vec3b v2 = src2.at<Vec3b>(pt2);
                for(int c = 0; c < 3; c++) {
                    double diff = v1[c] - v2[c];
                    sum += diff * diff;
                }
            }
        }
    }

    return sqrt(sum);
}

// Calculate gradient
void compute_gradient(const Mat& src, Mat& dx, Mat& dy) {
    dx = Mat::zeros(src.size(), CV_32F);
    dy = Mat::zeros(src.size(), CV_32F);

    #pragma omp parallel for
    for(int i = 1; i < src.rows-1; i++) {
        for(int j = 1; j < src.cols-1; j++) {
            dx.at<float>(i,j) = (float)(src.at<uchar>(i,j+1) - src.at<uchar>(i,j-1)) / 2.0f;
            dy.at<float>(i,j) = (float)(src.at<uchar>(i+1,j) - src.at<uchar>(i-1,j)) / 2.0f;
        }
    }
}

} // anonymous namespace

Mat diffusion_inpaint(
    const Mat& src,
    const Mat& mask,
    int radius,
    int num_iterations) {

    Mat result = src.clone();
    Mat mask_float;
    mask.convertTo(mask_float, CV_32F, 1.0/255.0);

    // Iterative diffusion
    for(int iter = 0; iter < num_iterations; iter++) {
        Mat next = result.clone();

        #pragma omp parallel for
        for(int i = radius; i < result.rows-radius; i++) {
            for(int j = radius; j < result.cols-radius; j++) {
                if(mask.at<uchar>(i,j) > 0) {
                    Vec3f sum(0,0,0);
                    float weight_sum = 0;

                    // Diffuse within neighborhood
                    for(int di = -radius; di <= radius; di++) {
                        for(int dj = -radius; dj <= radius; dj++) {
                            if(di == 0 && dj == 0) continue;

                            Point pt(j+dj, i+di);
                            if(mask.at<uchar>(pt) == 0) {
                                float w = 1.0f / (abs(di) + abs(dj));
                                sum += Vec3f(result.at<Vec3b>(pt)) * w;
                                weight_sum += w;
                            }
                        }
                    }

                    if(weight_sum > EPSILON) {
                        sum /= weight_sum;
                        next.at<Vec3b>(i,j) = Vec3b(sum);
                    }
                }
            }
        }

        result = next;
    }

    return result;
}

Mat patch_match_inpaint(
    const Mat& src,
    const Mat& mask,
    int patch_size,
    int search_area) {

    Mat result = src.clone();
    int half_patch = patch_size / 2;

    // Get points to be inpainted
    vector<Point> inpaint_points;
    for(int i = half_patch; i < mask.rows-half_patch; i++) {
        for(int j = half_patch; j < mask.cols-half_patch; j++) {
            if(mask.at<uchar>(i,j) > 0) {
                inpaint_points.push_back(Point(j,i));
            }
        }
    }

    // Find best matching patch for each point to be inpainted
    #pragma omp parallel for
    for(int k = 0; k < static_cast<int>(inpaint_points.size()); k++) {
        Point p = inpaint_points[k];
        double min_dist = numeric_limits<double>::max();
        Point best_match;

        // Search for best match in the search area
        for(int i = max(half_patch, p.y-search_area);
            i < min(src.rows-half_patch, p.y+search_area); i++) {
            for(int j = max(half_patch, p.x-search_area);
                j < min(src.cols-half_patch, p.x+search_area); j++) {
                if(mask.at<uchar>(i,j) == 0) {
                    double dist = compute_patch_similarity(
                        src, src, p, Point(j,i), patch_size);
                    if(dist < min_dist) {
                        min_dist = dist;
                        best_match = Point(j,i);
                    }
                }
            }
        }

        // Copy the best matching patch
        if(min_dist < numeric_limits<double>::max()) {
            for(int di = -half_patch; di <= half_patch; di++) {
                for(int dj = -half_patch; dj <= half_patch; dj++) {
                    Point src_pt = best_match + Point(dj,di);
                    Point dst_pt = p + Point(dj,di);
                    if(mask.at<uchar>(dst_pt) > 0) {
                        result.at<Vec3b>(dst_pt) = src.at<Vec3b>(src_pt);
                    }
                }
            }
        }
    }

    return result;
}

Mat fast_marching_inpaint(
    const Mat& src,
    const Mat& mask,
    int radius) {

    Mat result = src.clone();
    Mat known = (mask == 0);
    Mat dx, dy;

    // Calculate gradient
    compute_gradient(src, dx, dy);

    // Use priority queue to store boundary points
    // Define a custom comparator for the priority queue that compares the first element
    auto comp = [](const pair<float, Point>& a, const pair<float, Point>& b) {
        return a.first < b.first;
    };

    // Create the priority queue with the custom comparator
    priority_queue<pair<float, Point>, vector<pair<float, Point>>, decltype(comp)> boundary(comp);

    // Initialize boundary points
    for(int i = radius; i < src.rows-radius; i++) {
        for(int j = radius; j < src.cols-radius; j++) {
            if(mask.at<uchar>(i,j) > 0) {
                bool is_boundary = false;
                for(int di = -1; di <= 1; di++) {
                    for(int dj = -1; dj <= 1; dj++) {
                        if(known.at<uchar>(i+di,j+dj) > 0) {
                            is_boundary = true;
                            break;
                        }
                    }
                    if(is_boundary) break;
                }
                if(is_boundary) {
                    float priority = abs(dx.at<float>(i,j)) + abs(dy.at<float>(i,j));
                    boundary.push(make_pair(priority, Point(j,i)));
                }
            }
        }
    }

    // Fast marching inpainting
    while(!boundary.empty()) {
        Point p = boundary.top().second;
        boundary.pop();

        if(known.at<uchar>(p) > 0) continue;

        // Calculate inpainting value
        Vec3f sum(0,0,0);
        float weight_sum = 0;

        for(int di = -radius; di <= radius; di++) {
            for(int dj = -radius; dj <= radius; dj++) {
                Point pt(p.x+dj, p.y+di);
                if(known.at<uchar>(pt) > 0) {
                    float w = 1.0f / (abs(di) + abs(dj));
                    sum += Vec3f(result.at<Vec3b>(pt)) * w;
                    weight_sum += w;
                }
            }
        }

        if(weight_sum > EPSILON) {
            sum /= weight_sum;
            result.at<Vec3b>(p) = Vec3b(sum);
            known.at<uchar>(p) = 255;

            // Update boundary
            for(int di = -1; di <= 1; di++) {
                for(int dj = -1; dj <= 1; dj++) {
                    Point pt(p.x+dj, p.y+di);
                    if(mask.at<uchar>(pt) > 0 && known.at<uchar>(pt) == 0) {
                        float priority = abs(dx.at<float>(pt.y,pt.x)) +
                                      abs(dy.at<float>(pt.y,pt.x));
                        boundary.push(make_pair(priority, pt));
                    }
                }
            }
        }
    }

    return result;
}

Mat texture_synthesis_inpaint(
    const Mat& src,
    const Mat& mask,
    int patch_size,
    int overlap) {

    Mat result = src.clone();
    int half_patch = patch_size / 2;

    // Get all patches to be inpainted
    vector<Rect> patches;
    for(int i = half_patch; i < src.rows-half_patch; i += patch_size-overlap) {
        for(int j = half_patch; j < src.cols-half_patch; j += patch_size-overlap) {
            if(mask.at<uchar>(i,j) > 0) {
                patches.push_back(Rect(j-half_patch, i-half_patch,
                                     patch_size, patch_size));
            }
        }
    }

    // Inpaint each patch
    #pragma omp parallel for
    for(int k = 0; k < static_cast<int>(patches.size()); k++) {
        Rect patch = patches[k];
        double min_error = numeric_limits<double>::max();
        Mat best_patch;

        // Search for best matching patch in the source image
        for(int i = half_patch; i < src.rows-half_patch; i += patch_size/2) {
            for(int j = half_patch; j < src.cols-half_patch; j += patch_size/2) {
                if(mask.at<uchar>(i,j) == 0) {
                    Rect r(j-half_patch, i-half_patch, patch_size, patch_size);
                    Mat candidate = src(r);

                    // Calculate error for overlapping regions
                    double error = 0;
                    if(patch.x > 0) {  // Left overlap
                        Mat left_src = result(Rect(patch.x-overlap, patch.y,
                                                 overlap, patch_size));
                        Mat left_cand = candidate(Rect(0, 0, overlap, patch_size));
                        error += norm(left_src, left_cand);
                    }
                    if(patch.y > 0) {  // Top overlap
                        Mat top_src = result(Rect(patch.x, patch.y-overlap,
                                                patch_size, overlap));
                        Mat top_cand = candidate(Rect(0, 0, patch_size, overlap));
                        error += norm(top_src, top_cand);
                    }

                    if(error < min_error) {
                        min_error = error;
                        best_patch = candidate.clone();
                    }
                }
            }
        }

        // Copy the best matching patch to the result
        if(!best_patch.empty()) {
            best_patch.copyTo(result(patch));
        }
    }

    return result;
}

Mat structure_propagation_inpaint(
    const Mat& src,
    const Mat& mask,
    int patch_size,
    int num_iterations) {

    Mat result = src.clone();
    Mat confidence = (mask == 0);
    confidence.convertTo(confidence, CV_32F, 1.0/255.0);
    int half_patch = patch_size / 2;

    // Calculate structure tensor
    Mat dx, dy;
    compute_gradient(src, dx, dy);
    Mat tensor = Mat::zeros(src.size(), CV_32FC3);

    #pragma omp parallel for
    for(int i = 1; i < src.rows-1; i++) {
        for(int j = 1; j < src.cols-1; j++) {
            float ix = dx.at<float>(i,j);
            float iy = dy.at<float>(i,j);
            tensor.at<Vec3f>(i,j) = Vec3f(ix*ix, ix*iy, iy*iy);
        }
    }

    // Iterative inpainting
    for(int iter = 0; iter < num_iterations; iter++) {
        // Find boundary point with highest confidence
        Point max_pt;
        float max_conf = -1;

        for(int i = half_patch; i < src.rows-half_patch; i++) {
            for(int j = half_patch; j < src.cols-half_patch; j++) {
                if(mask.at<uchar>(i,j) > 0) {
                    bool is_boundary = false;
                    float conf_sum = 0;
                    for(int di = -1; di <= 1; di++) {
                        for(int dj = -1; dj <= 1; dj++) {
                            Point pt(j+dj, i+di);
                            if(confidence.at<float>(pt) > 0) {
                                is_boundary = true;
                                conf_sum += confidence.at<float>(pt);
                            }
                        }
                    }
                    if(is_boundary && conf_sum > max_conf) {
                        max_conf = conf_sum;
                        max_pt = Point(j,i);
                    }
                }
            }
        }

        if(max_conf < 0) break;

        // Perform structure propagation at boundary point
        Vec3f t = tensor.at<Vec3f>(max_pt);
        float angle = 0.5f * atan2(2*t[1], t[0]-t[2]);
        Point2f dir(cos(angle), sin(angle));

        // Find best matching patch
        double min_dist = numeric_limits<double>::max();
        Point best_match;

        for(int i = half_patch; i < src.rows-half_patch; i++) {
            for(int j = half_patch; j < src.cols-half_patch; j++) {
                if(mask.at<uchar>(i,j) == 0) {
                    Point pt(j,i);
                    Vec3f t2 = tensor.at<Vec3f>(pt);
                    float angle2 = 0.5f * atan2(2*t2[1], t2[0]-t2[2]);
                    Point2f dir2(cos(angle2), sin(angle2));

                    // Consider structure similarity
                    float dir_sim = abs(dir.dot(dir2));
                    if(dir_sim > 0.8f) {
                        double dist = compute_patch_similarity(
                            src, src, max_pt, pt, patch_size);
                        dist *= (2 - dir_sim);  // Structure weight

                        if(dist < min_dist) {
                            min_dist = dist;
                            best_match = pt;
                        }
                    }
                }
            }
        }

        // Copy best matching patch
        if(min_dist < numeric_limits<double>::max()) {
            for(int di = -half_patch; di <= half_patch; di++) {
                for(int dj = -half_patch; dj <= half_patch; dj++) {
                    Point src_pt = best_match + Point(dj,di);
                    Point dst_pt = max_pt + Point(dj,di);
                    if(mask.at<uchar>(dst_pt) > 0) {
                        result.at<Vec3b>(dst_pt) = src.at<Vec3b>(src_pt);
                        confidence.at<float>(dst_pt) =
                            confidence.at<float>(src_pt) * 0.9f;
                    }
                }
            }
        }
    }

    return result;
}

Mat patchmatch_inpaint(
    const Mat& src,
    const Mat& mask,
    int patch_size,
    int num_iterations) {

    Mat result = src.clone();
    int half_patch = patch_size / 2;

    // Initialize random matching
    RNG rng;
    Mat offsets(mask.size(), CV_32SC2);
    for(int i = 0; i < mask.rows; i++) {
        for(int j = 0; j < mask.cols; j++) {
            if(mask.at<uchar>(i,j) > 0) {
                int dx = rng.uniform(0, src.cols);
                int dy = rng.uniform(0, src.rows);
                offsets.at<Vec2i>(i,j) = Vec2i(dx-j, dy-i);
            }
        }
    }

    // Iterative optimization
    for(int iter = 0; iter < num_iterations; iter++) {
        // Propagation
        for(int i = 0; i < mask.rows; i++) {
            for(int j = 0; j < mask.cols; j++) {
                if(mask.at<uchar>(i,j) > 0) {
                    // Check matching of adjacent pixels
                    vector<Point> neighbors = {
                        Point(j-1,i), Point(j+1,i),
                        Point(j,i-1), Point(j,i+1)
                    };

                    for(const auto& n : neighbors) {
                        if(n.x >= 0 && n.x < mask.cols &&
                           n.y >= 0 && n.y < mask.rows) {
                            Vec2i offset = offsets.at<Vec2i>(n);
                            Point match(j+offset[0], i+offset[1]);

                            if(match.x >= 0 && match.x < src.cols &&
                               match.y >= 0 && match.y < src.rows) {
                                double dist = compute_patch_similarity(
                                    src, src, Point(j,i), match, patch_size);
                                Vec2i currOffset = offsets.at<Vec2i>(i,j);
                                Point currMatch(j+currOffset[0], i+currOffset[1]);
                                if(dist < compute_patch_similarity(
                                    src, src, Point(j,i), currMatch, patch_size)) {
                                    offsets.at<Vec2i>(i,j) = offset;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Random search
        for(int i = 0; i < mask.rows; i++) {
            for(int j = 0; j < mask.cols; j++) {
                if(mask.at<uchar>(i,j) > 0) {
                    int searchRadius = src.cols;
                    while(searchRadius > 1) {
                        int dx = rng.uniform(-searchRadius, searchRadius);
                        int dy = rng.uniform(-searchRadius, searchRadius);
                        Point match(j+dx, i+dy);

                        if(match.x >= 0 && match.x < src.cols &&
                           match.y >= 0 && match.y < src.rows) {
                            double dist = compute_patch_similarity(
                                src, src, Point(j,i), match, patch_size);
                            Vec2i currOffset = offsets.at<Vec2i>(i,j);
                            Point currMatch(j+currOffset[0], i+currOffset[1]);
                            if(dist < compute_patch_similarity(
                                src, src, Point(j,i), currMatch, patch_size)) {
                                offsets.at<Vec2i>(i,j) = Vec2i(dx, dy);
                            }
                        }
                        searchRadius /= 2;
                    }
                }
            }
        }
    }

    // Apply best matching
    for(int i = 0; i < mask.rows; i++) {
        for(int j = 0; j < mask.cols; j++) {
            if(mask.at<uchar>(i,j) > 0) {
                Vec2i offset = offsets.at<Vec2i>(i,j);
                Point match(j+offset[0], i+offset[1]);
                if(match.x >= 0 && match.x < src.cols &&
                   match.y >= 0 && match.y < src.rows) {
                    result.at<Vec3b>(i,j) = src.at<Vec3b>(match);
                }
            }
        }
    }

    return result;
}

vector<Mat> video_inpaint(
    const vector<Mat>& frames,
    const vector<Mat>& masks,
    int patch_size,
    int num_iterations) {

    vector<Mat> results;
    for(const auto& frame : frames) {
        results.push_back(frame.clone());
    }
    int half_patch = patch_size / 2;

    // Calculate optical flow field
    vector<Mat> flow_forward, flow_backward;
    for(size_t i = 0; i < frames.size()-1; i++) {
        Mat flow;
        calcOpticalFlowFarneback(frames[i], frames[i+1], flow,
                               0.5, 3, 15, 3, 5, 1.2, 0);
        flow_forward.push_back(flow);
    }

    for(size_t i = frames.size()-1; i > 0; i--) {
        Mat flow;
        calcOpticalFlowFarneback(frames[i], frames[i-1], flow,
                               0.5, 3, 15, 3, 5, 1.2, 0);
        flow_backward.push_back(flow);
    }

    // Iterative inpainting
    for(int iter = 0; iter < num_iterations; iter++) {
        for(size_t t = 0; t < frames.size(); t++) {
            // Get temporal neighborhood
            vector<Mat> temporal_patches;
            if(t > 0) {
                Mat map1, map2;
                Mat& flow = flow_backward[t-1];
                convertMaps(flow, Mat(), map1, map2, CV_32FC1);
                Mat warped;
                remap(results[t-1], warped, map1, map2, INTER_LINEAR);
                temporal_patches.push_back(warped);
            }
            if(t < frames.size()-1) {
                Mat map1, map2;
                Mat& flow = flow_forward[t];
                convertMaps(flow, Mat(), map1, map2, CV_32FC1);
                Mat warped;
                remap(results[t+1], warped, map1, map2, INTER_LINEAR);
                temporal_patches.push_back(warped);
            }

            // Inpaint current frame
            for(int i = half_patch; i < frames[t].rows-half_patch; i++) {
                for(int j = half_patch; j < frames[t].cols-half_patch; j++) {
                    if(masks[t].at<uchar>(i,j) > 0) {
                        double min_dist = numeric_limits<double>::max();
                        Point best_match;

                        // Spatial matching
                        for(int di = -half_patch; di <= half_patch; di++) {
                            for(int dj = -half_patch; dj <= half_patch; dj++) {
                                if(masks[t].at<uchar>(i+di,j+dj) == 0) {
                                    double dist = compute_patch_similarity(
                                        results[t], results[t],
                                        Point(j,i), Point(j+dj,i+di), patch_size);
                                    if(dist < min_dist) {
                                        min_dist = dist;
                                        best_match = Point(j+dj,i+di);
                                    }
                                }
                            }
                        }

                        // Temporal matching
                        for(const auto& patch : temporal_patches) {
                            for(int di = -half_patch; di <= half_patch; di++) {
                                for(int dj = -half_patch; dj <= half_patch; dj++) {
                                    Point pt(j+dj, i+di);
                                    if(pt.x >= 0 && pt.x < patch.cols &&
                                       pt.y >= 0 && pt.y < patch.rows) {
                                        double dist = compute_patch_similarity(
                                            results[t], patch,
                                            Point(j,i), pt, patch_size);
                                        if(dist < min_dist) {
                                            min_dist = dist;
                                            best_match = pt;
                                        }
                                    }
                                }
                            }
                        }

                        // Apply best matching
                        if(min_dist < numeric_limits<double>::max()) {
                            results[t].at<Vec3b>(i,j) =
                                results[t].at<Vec3b>(best_match);
                        }
                    }
                }
            }
        }
    }

    return results;
}

} // namespace ip101