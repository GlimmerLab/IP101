#include <basic/object_detection.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>  // 添加 ml 模块支持 SVM
#include <omp.h>
#include <numeric>  // 添加 numeric 支持 iota

namespace ip101 {

using namespace cv;
using namespace std;

namespace {
// Internal constants
constexpr int CACHE_LINE = 64;    // CPU cache line size (bytes)
constexpr int BLOCK_SIZE = 16;    // Block processing size

// HOG feature extractor
class HOGExtractor {
public:
    HOGExtractor(Size win_size = Size(64, 128),
                Size block_size = Size(16, 16),
                Size block_stride = Size(8, 8),
                Size cell_size = Size(8, 8),
                int nbins = 9)
        : win_size_(win_size),
          block_size_(block_size),
          block_stride_(block_stride),
          cell_size_(cell_size),
          nbins_(nbins) {
        // Calculate HOG feature dimension
        Size n_cells(win_size.width / cell_size.width,
                    win_size.height / cell_size.height);
        Size n_blocks((n_cells.width - block_size.width / cell_size.width) /
                     (block_stride.width / cell_size.width) + 1,
                     (n_cells.height - block_size.height / cell_size.height) /
                     (block_stride.height / cell_size.height) + 1);
        feature_dim_ = n_blocks.width * n_blocks.height *
                      block_size.width * block_size.height *
                      nbins / (cell_size.width * cell_size.height);
    }

    // Calculate image gradients
    void computeGradients(const Mat& img, Mat& magnitude, Mat& angle) const {
        magnitude = Mat::zeros(img.size(), CV_32F);
        angle = Mat::zeros(img.size(), CV_32F);

        #pragma omp parallel for
        for (int y = 1; y < img.rows - 1; y++) {
            for (int x = 1; x < img.cols - 1; x++) {
                // Calculate x-direction gradient
                float gx = static_cast<float>(img.at<uchar>(y, x + 1) - img.at<uchar>(y, x - 1));
                // Calculate y-direction gradient
                float gy = static_cast<float>(img.at<uchar>(y + 1, x) - img.at<uchar>(y - 1, x));

                // Calculate gradient magnitude
                magnitude.at<float>(y, x) = std::sqrt(gx * gx + gy * gy);
                // Calculate gradient direction (angle)
                angle.at<float>(y, x) = static_cast<float>(std::atan2(gy, gx) * 180.0 / CV_PI);
                if (angle.at<float>(y, x) < 0) {
                    angle.at<float>(y, x) += 180.0f;
                }
            }
        }
    }

    // Calculate cell histogram
    void computeCellHistogram(const Mat& magnitude, const Mat& angle,
                            vector<vector<vector<float>>>& cell_hists) const {
        Size n_cells(win_size_.width / cell_size_.width,
                    win_size_.height / cell_size_.height);

        #pragma omp parallel for
        for (int y = 0; y < magnitude.rows; y++) {
            for (int x = 0; x < magnitude.cols; x++) {
                float mag = magnitude.at<float>(y, x);
                float ang = angle.at<float>(y, x);

                // Calculate bin index
                float bin_width = 180.0f / nbins_;
                int bin = static_cast<int>(ang / bin_width);
                int next_bin = (bin + 1) % nbins_;
                float alpha = (ang - bin * bin_width) / bin_width;

                // Calculate cell index
                int cell_x = x / cell_size_.width;
                int cell_y = y / cell_size_.height;

                if (cell_x < n_cells.width && cell_y < n_cells.height) {
                    #pragma omp atomic
                    cell_hists[cell_y][cell_x][bin] += mag * (1 - alpha);
                    #pragma omp atomic
                    cell_hists[cell_y][cell_x][next_bin] += mag * alpha;
                }
            }
        }
    }

    // Calculate block features and normalize
    void computeBlockFeatures(const vector<vector<vector<float>>>& cell_hists,
                            vector<float>& features) const {
        Size n_cells(win_size_.width / cell_size_.width,
                    win_size_.height / cell_size_.height);
        Size n_blocks((n_cells.width - block_size_.width / cell_size_.width) /
                     (block_stride_.width / cell_size_.width) + 1,
                     (n_cells.height - block_size_.height / cell_size_.height) /
                     (block_stride_.height / cell_size_.height) + 1);

        for (int by = 0; by <= n_cells.height - block_size_.height / cell_size_.height;
             by += block_stride_.height / cell_size_.height) {
            for (int bx = 0; bx <= n_cells.width - block_size_.width / cell_size_.width;
                 bx += block_stride_.width / cell_size_.width) {
                vector<float> block_features;
                float norm = 0;

                // Collect all cell histograms in the block
                for (int cy = by; cy < by + block_size_.height / cell_size_.height; cy++) {
                    for (int cx = bx; cx < bx + block_size_.width / cell_size_.width; cx++) {
                        block_features.insert(block_features.end(),
                                           cell_hists[cy][cx].begin(),
                                           cell_hists[cy][cx].end());
                        for (float val : cell_hists[cy][cx]) {
                            norm += val * val;
                        }
                    }
                }

                // L2-Norm normalization
                norm = std::sqrt(norm + 1e-5f);
                for (float& val : block_features) {
                    val /= norm;
                }

                features.insert(features.end(),
                              block_features.begin(),
                              block_features.end());
            }
        }
    }

    vector<float> compute(const Mat& img) const {
        // Resize image
        Mat resized;
        resize(img, resized, win_size_);

        // Calculate gradients
        Mat magnitude, angle;
        computeGradients(resized, magnitude, angle);

        // Calculate cell histograms
        Size n_cells(win_size_.width / cell_size_.width,
                    win_size_.height / cell_size_.height);
        vector<vector<vector<float>>> cell_hists(n_cells.height,
            vector<vector<float>>(n_cells.width, vector<float>(nbins_, 0)));
        computeCellHistogram(magnitude, angle, cell_hists);

        // Calculate block features and normalize
        vector<float> features;
        features.reserve(feature_dim_);
        computeBlockFeatures(cell_hists, features);

        return features;
    }

private:
    Size win_size_;
    Size block_size_;
    Size block_stride_;
    Size cell_size_;
    int nbins_;
    int feature_dim_;
};

// Haar feature extractor
class HaarExtractor {
public:
    HaarExtractor() {
        // Load pre-trained Haar cascade classifier
        face_cascade_.load("haarcascade_frontalface_alt.xml");
    }

    vector<Rect> detect(const Mat& img, double scale_factor = 1.1, int min_neighbors = 3) {
        vector<Rect> faces;
        face_cascade_.detectMultiScale(img, faces, scale_factor, min_neighbors);
        return faces;
    }

private:
    CascadeClassifier face_cascade_;
};

// Calculate IoU between two rectangles
float compute_iou(const Rect& a, const Rect& b) {
    int x1 = max(a.x, b.x);
    int y1 = max(a.y, b.y);
    int x2 = min(a.x + a.width, b.x + b.width);
    int y2 = min(a.y + a.height, b.y + b.height);

    if (x1 >= x2 || y1 >= y2) return 0.0f;

    float intersection_area = static_cast<float>((x2 - x1) * (y2 - y1));
    float union_area = static_cast<float>(a.width * a.height + b.width * b.height - intersection_area);

    return intersection_area / union_area;
}

} // anonymous namespace

vector<DetectionResult> sliding_window_detect(
    const Mat& src,
    const Size& window_size,
    int stride,
    float threshold) {

    vector<DetectionResult> results;
    HOGExtractor hog(window_size);

    // Load pre-trained SVM model
    Ptr<ml::SVM> svm = ml::SVM::load("pedestrian_svm.xml");

    #pragma omp parallel for
    for (int y = 0; y <= src.rows - window_size.height; y += stride) {
        for (int x = 0; x <= src.cols - window_size.width; x += stride) {
            // Extract window
            Mat window = src(Rect(x, y, window_size.width, window_size.height));

            // Calculate HOG features
            vector<float> features = hog.compute(window);

            // SVM prediction
            Mat feature_mat(1, static_cast<int>(features.size()), CV_32F);
            memcpy(feature_mat.data, features.data(), features.size() * sizeof(float));
            float score = svm->predict(feature_mat, noArray(), ml::StatModel::RAW_OUTPUT);

            if (score > threshold) {
                DetectionResult det;
                det.bbox = Rect(x, y, window_size.width, window_size.height);
                det.confidence = score;
                det.class_id = 1;  // pedestrian class
                det.label = "pedestrian";

                #pragma omp critical
                results.push_back(det);
            }
        }
    }

    return results;
}

vector<DetectionResult> hog_svm_detect(const Mat& src, float threshold) {
    vector<DetectionResult> results;

    // Use OpenCV built-in HOG detector
    HOGDescriptor hog;
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

    // Detect pedestrians
    vector<Rect> found;
    vector<double> weights;
    hog.detectMultiScale(src, found, weights,
                        0,              // hit threshold
                        Size(8,8),      // win stride
                        Size(32,32),    // padding
                        1.05,           // scale
                        threshold);     // final threshold

    // Convert result format
    for (size_t i = 0; i < found.size(); i++) {
        DetectionResult det;
        det.bbox = found[i];
        det.confidence = static_cast<float>(weights[i]);
        det.class_id = 1;  // pedestrian class
        det.label = "pedestrian";
        results.push_back(det);
    }

    return results;
}

vector<DetectionResult> haar_face_detect(const Mat& src, float threshold) {
    vector<DetectionResult> results;

    // Create Haar feature extractor
    HaarExtractor haar;

    // Detect faces
    vector<Rect> faces = haar.detect(src, 1.1, 3);

    // Convert result format
    for (const auto& face : faces) {
        DetectionResult det;
        det.bbox = face;
        det.confidence = 1.0f;  // Haar classifier has no confidence output
        det.class_id = 2;  // face class
        det.label = "face";
        results.push_back(det);
    }

    return results;
}

vector<int> nms(const vector<Rect>& boxes,
                const vector<float>& scores,
                float iou_threshold) {

    vector<int> indices(boxes.size());
    std::iota(indices.begin(), indices.end(), 0);  // Fill with 0, 1, 2, ...

    // Sort by confidence in descending order
    sort(indices.begin(), indices.end(),
         [&scores](int a, int b) { return scores[a] > scores[b]; });

    vector<int> keep;
    while (!indices.empty()) {
        int curr = indices[0];
        keep.push_back(curr);

        vector<int> tmp;
        for (size_t i = 1; i < indices.size(); i++) {
            float iou = compute_iou(boxes[curr], boxes[indices[i]]);
            if (iou <= iou_threshold) {
                tmp.push_back(indices[i]);
            }
        }
        indices = tmp;
    }

    return keep;
}

vector<DetectionResult> track_objects(
    const Mat& src,
    const Mat& prev,
    const vector<DetectionResult>& prev_boxes) {

    vector<DetectionResult> curr_boxes;

    // Calculate optical flow
    vector<Point2f> prev_points, curr_points;
    for (const auto& det : prev_boxes) {
        prev_points.push_back(Point2f(static_cast<float>(det.bbox.x + det.bbox.width/2),
                                    static_cast<float>(det.bbox.y + det.bbox.height/2)));
    }

    vector<uchar> status;
    vector<float> err;
    calcOpticalFlowPyrLK(prev, src, prev_points, curr_points, status, err);

    // Update detection box positions
    for (size_t i = 0; i < prev_boxes.size(); i++) {
        if (status[i]) {
            DetectionResult det = prev_boxes[i];
            float dx = curr_points[i].x - prev_points[i].x;
            float dy = curr_points[i].y - prev_points[i].y;
            det.bbox.x += static_cast<int>(dx);
            det.bbox.y += static_cast<int>(dy);
            curr_boxes.push_back(det);
        }
    }

    return curr_boxes;
}

Mat draw_detections(const Mat& src,
                   const vector<DetectionResult>& detections) {

    Mat img_display = src.clone();
    if (img_display.channels() == 1) {
        cvtColor(img_display, img_display, COLOR_GRAY2BGR);
    }

    for (const auto& det : detections) {
        Scalar color;
        if (det.class_id == 1) {  // pedestrian
            color = Scalar(0, 255, 0);
        } else if (det.class_id == 2) {  // face
            color = Scalar(0, 0, 255);
        } else {
            color = Scalar(255, 0, 0);
        }

        // Draw bounding box
        rectangle(img_display, det.bbox, color, 2);

        // Draw label and confidence
        string label = format("%s %.2f", det.label.c_str(), det.confidence);
        int baseline = 0;
        Size text_size = getTextSize(label, FONT_HERSHEY_SIMPLEX,
                                   0.5, 1, &baseline);
        rectangle(img_display,
                 Point(det.bbox.x, det.bbox.y - text_size.height - 5),
                 Point(det.bbox.x + text_size.width, det.bbox.y),
                 color, -1);
        putText(img_display, label,
                Point(det.bbox.x, det.bbox.y - 5),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
    }

    return img_display;
}

} // namespace ip101