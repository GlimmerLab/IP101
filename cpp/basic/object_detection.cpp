#include "object_detection.hpp"
#include <omp.h>

namespace ip101 {

using namespace cv;
using namespace std;

namespace {
// 内部常量定义
constexpr int CACHE_LINE = 64;    // CPU缓存行大小(字节)
constexpr int BLOCK_SIZE = 16;    // 分块处理大小

// HOG特征提取器
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
        // 计算HOG特征维度
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

    // 计算图像梯度
    void computeGradients(const Mat& img, Mat& magnitude, Mat& angle) const {
        magnitude = Mat::zeros(img.size(), CV_32F);
        angle = Mat::zeros(img.size(), CV_32F);

        #pragma omp parallel for collapse(2)
        for (int y = 1; y < img.rows - 1; y++) {
            for (int x = 1; x < img.cols - 1; x++) {
                // 计算x方向梯度
                float gx = img.at<uchar>(y, x + 1) - img.at<uchar>(y, x - 1);
                // 计算y方向梯度
                float gy = img.at<uchar>(y + 1, x) - img.at<uchar>(y - 1, x);

                // 计算梯度幅值
                magnitude.at<float>(y, x) = std::sqrt(gx * gx + gy * gy);
                // 计算梯度方向（角度）
                angle.at<float>(y, x) = std::atan2(gy, gx) * 180.0 / CV_PI;
                if (angle.at<float>(y, x) < 0) {
                    angle.at<float>(y, x) += 180.0;
                }
            }
        }
    }

    // 计算cell直方图
    void computeCellHistogram(const Mat& magnitude, const Mat& angle,
                            vector<vector<vector<float>>>& cell_hists) const {
        Size n_cells(win_size_.width / cell_size_.width,
                    win_size_.height / cell_size_.height);

        #pragma omp parallel for collapse(2)
        for (int y = 0; y < magnitude.rows; y++) {
            for (int x = 0; x < magnitude.cols; x++) {
                float mag = magnitude.at<float>(y, x);
                float ang = angle.at<float>(y, x);

                // 计算bin索引
                float bin_width = 180.0f / nbins_;
                int bin = static_cast<int>(ang / bin_width);
                int next_bin = (bin + 1) % nbins_;
                float alpha = (ang - bin * bin_width) / bin_width;

                // 计算cell索引
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

    // 计算block特征并归一化
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

                // 收集block中的所有cell直方图
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

                // L2-Norm归一化
                norm = std::sqrt(norm + 1e-5);
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
        // 调整图像大小
        Mat resized;
        resize(img, resized, win_size_);

        // 计算梯度
        Mat magnitude, angle;
        computeGradients(resized, magnitude, angle);

        // 计算cell直方图
        Size n_cells(win_size_.width / cell_size_.width,
                    win_size_.height / cell_size_.height);
        vector<vector<vector<float>>> cell_hists(n_cells.height,
            vector<vector<float>>(n_cells.width, vector<float>(nbins_, 0)));
        computeCellHistogram(magnitude, angle, cell_hists);

        // 计算block特征并归一化
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

// Haar特征提取器
class HaarExtractor {
public:
    HaarExtractor() {
        // 加载预训练的Haar级联分类器
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

// 计算两个矩形的IoU
float compute_iou(const Rect& a, const Rect& b) {
    int x1 = max(a.x, b.x);
    int y1 = max(a.y, b.y);
    int x2 = min(a.x + a.width, b.x + b.width);
    int y2 = min(a.y + a.height, b.y + b.height);

    if (x1 >= x2 || y1 >= y2) return 0.0f;

    float intersection_area = (x2 - x1) * (y2 - y1);
    float union_area = a.width * a.height + b.width * b.height - intersection_area;

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

    // 加载预训练的SVM模型
    Ptr<SVM> svm = Algorithm::load<SVM>("pedestrian_svm.xml");

    #pragma omp parallel for collapse(2)
    for (int y = 0; y <= src.rows - window_size.height; y += stride) {
        for (int x = 0; x <= src.cols - window_size.width; x += stride) {
            // 提取窗口
            Mat window = src(Rect(x, y, window_size.width, window_size.height));

            // 计算HOG特征
            vector<float> features = hog.compute(window);

            // SVM预测
            Mat feature_mat(1, (int)features.size(), CV_32F);
            memcpy(feature_mat.data, features.data(), features.size() * sizeof(float));
            float score = svm->predict(feature_mat, noArray(), StatModel::RAW_OUTPUT);

            if (score > threshold) {
                DetectionResult det;
                det.bbox = Rect(x, y, window_size.width, window_size.height);
                det.confidence = score;
                det.class_id = 1;  // 行人类别
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

    // 使用OpenCV内置的HOG检测器
    HOGDescriptor hog;
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

    // 检测行人
    vector<Rect> found;
    vector<double> weights;
    hog.detectMultiScale(src, found, weights,
                        0,              // hit threshold
                        Size(8,8),      // win stride
                        Size(32,32),    // padding
                        1.05,           // scale
                        threshold);     // final threshold

    // 转换结果格式
    for (size_t i = 0; i < found.size(); i++) {
        DetectionResult det;
        det.bbox = found[i];
        det.confidence = weights[i];
        det.class_id = 1;  // 行人类别
        det.label = "pedestrian";
        results.push_back(det);
    }

    return results;
}

vector<DetectionResult> haar_face_detect(const Mat& src, float threshold) {
    vector<DetectionResult> results;

    // 创建Haar特征提取器
    HaarExtractor haar;

    // 检测人脸
    vector<Rect> faces = haar.detect(src, 1.1, 3);

    // 转换结果格式
    for (const auto& face : faces) {
        DetectionResult det;
        det.bbox = face;
        det.confidence = 1.0f;  // Haar分类器没有置信度输出
        det.class_id = 2;  // 人脸类别
        det.label = "face";
        results.push_back(det);
    }

    return results;
}

vector<int> nms(const vector<Rect>& boxes,
                const vector<float>& scores,
                float iou_threshold) {

    vector<int> indices(boxes.size());
    iota(indices.begin(), indices.end(), 0);

    // 按置信度降序排序
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

    // 计算光流
    vector<Point2f> prev_points, curr_points;
    for (const auto& det : prev_boxes) {
        prev_points.push_back(Point2f(det.bbox.x + det.bbox.width/2,
                                    det.bbox.y + det.bbox.height/2));
    }

    vector<uchar> status;
    vector<float> err;
    calcOpticalFlowPyrLK(prev, src, prev_points, curr_points, status, err);

    // 更新检测框位置
    for (size_t i = 0; i < prev_boxes.size(); i++) {
        if (status[i]) {
            DetectionResult det = prev_boxes[i];
            float dx = curr_points[i].x - prev_points[i].x;
            float dy = curr_points[i].y - prev_points[i].y;
            det.bbox.x += dx;
            det.bbox.y += dy;
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
        if (det.class_id == 1) {  // 行人
            color = Scalar(0, 255, 0);
        } else if (det.class_id == 2) {  // 人脸
            color = Scalar(0, 0, 255);
        } else {
            color = Scalar(255, 0, 0);
        }

        // 绘制边界框
        rectangle(img_display, det.bbox, color, 2);

        // 绘制标签和置信度
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