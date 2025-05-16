# Object Detection Explorer Guide 🔍

> Object detection is like being a meticulous detective! We need to search for and locate specific objects in images, just like a detective searching for clues at a scene. Let's explore this challenging field of image processing together!

## Table of Contents
- [1. What is Object Detection?](#1-what-is-object-detection)
- [2. Sliding Window Detection](#2-sliding-window-detection)
- [3. HOG+SVM Detection](#3-hogsvm-detection)
- [4. Haar+AdaBoost Detection](#4-haaradaboost-detection)
- [5. Non-Maximum Suppression](#5-non-maximum-suppression)
- [6. Object Tracking](#6-object-tracking)
- [7. Code Implementation and Optimization](#7-code-implementation-and-optimization)
- [8. Applications and Best Practices](#8-applications-and-best-practices)
## 1. What is Object Detection?

Imagine you're an image detective, searching for "clues" in an image. Object detection is exactly that process, helping us with:

- 🔍 Locating object positions (finding "clue" locations)
- 📏 Determining object sizes (measuring "clue" dimensions)
- 🎯 Identifying object categories (determining "clue" types)
- 🔄 Tracking object motion (following "clue" changes)

## 2. Sliding Window Detection

### 2.1 Basic Principles

Sliding window is like a detective examining the scene grid by grid with a magnifying glass! It searches for objects by sliding windows of different sizes across the image.

Key steps:
1. Multi-scale pyramid
2. Window sliding
3. Feature extraction
4. Classification

### 2.2 Implementation Example

```cpp
// Sliding window detection implementation
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

            // Compute HOG features
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
```

```python
def sliding_window_detection(img_path, window_size=(64, 64), stride=32):
    """
    Sliding window detection implementation
    Parameters:
        img_path: Input image path
        window_size: Window size, default (64, 64)
        stride: Stride, default 32
    Returns:
        Detection result visualization
    """
    # Read image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")

    # Copy original image for drawing results
    result = img.copy()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Define simple object detection function (using edge detection as example)
    def detect_object(window):
        edges = cv2.Canny(window, 100, 200)
        return np.sum(edges) > 1000  # Simple threshold judgment

    # Sliding window detection
    for y in range(0, img.shape[0] - window_size[1], stride):
        for x in range(0, img.shape[1] - window_size[0], stride):
            # Extract window
            window = gray[y:y+window_size[1], x:x+window_size[0]]

            # Detect object
            if detect_object(window):
                # Draw detection box
                cv2.rectangle(result, (x, y),
                            (x + window_size[0], y + window_size[1]),
                            (0, 255, 0), 2)

    return result
```

## 3. HOG+SVM Detection

### 3.1 Algorithm Principles

HOG (Histogram of Oriented Gradients) features are like "texture clues" extracted by the detective, while SVM (Support Vector Machine) is like an experienced "judgment expert."

HOG feature calculation:
$$
\begin{aligned}
g_x &= I(x+1,y) - I(x-1,y) \\
g_y &= I(x,y+1) - I(x,y-1) \\
g &= \sqrt{g_x^2 + g_y^2} \\
\theta &= \arctan(\frac{g_y}{g_x})
\end{aligned}
$$

### 3.2 Implementation Example

```cpp
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

    // Compute image gradients
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

    // Compute cell histogram
    void computeCellHistogram(const Mat& magnitude, const Mat& angle,
                            vector<vector<vector<float>>>& cell_hists) const {
        Size n_cells(win_size_.width / cell_size_.width,
                    win_size_.height / cell_size_.height);

        #pragma omp parallel for collapse(2)
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

    // Compute block features and normalize
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

                // Collect cell histograms in block
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
        // Resize image
        Mat resized;
        resize(img, resized, win_size_);

        // Compute gradients
        Mat magnitude, angle;
        computeGradients(resized, magnitude, angle);

        // Compute cell histograms
        Size n_cells(win_size_.width / cell_size_.width,
                    win_size_.height / cell_size_.height);
        vector<vector<vector<float>>> cell_hists(n_cells.height,
            vector<vector<float>>(n_cells.width, vector<float>(nbins_, 0)));
        computeCellHistogram(magnitude, angle, cell_hists);

        // Compute block features and normalize
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
```

```python
class HOGExtractor:
    """HOG feature extractor"""
    def __init__(self, win_size=(64, 128), block_size=(16, 16),
                 block_stride=(8, 8), cell_size=(8, 8), nbins=9):
        self.win_size = win_size
        self.block_size = block_size
        self.block_stride = block_stride
        self.cell_size = cell_size
        self.nbins = nbins

        # Calculate feature dimension
        n_cells = (win_size[0] // cell_size[0], win_size[1] // cell_size[1])
        n_blocks = ((n_cells[0] - block_size[0] // cell_size[0]) // (block_stride[0] // cell_size[0]) + 1,
                   (n_cells[1] - block_size[1] // cell_size[1]) // (block_stride[1] // cell_size[1]) + 1)
        self.feature_dim = n_blocks[0] * n_blocks[1] * block_size[0] * block_size[1] * nbins // (cell_size[0] * cell_size[1])

    def compute_gradients(self, img):
        """Compute image gradients"""
        # Ensure image is grayscale
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Calculate x and y direction gradients
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)

        # Calculate gradient magnitude and direction
        magnitude = np.sqrt(gx**2 + gy**2)
        angle = np.arctan2(gy, gx) * 180 / np.pi
        angle[angle < 0] += 180

        return magnitude, angle

    def compute_cell_histogram(self, magnitude, angle):
        """Compute cell histogram"""
        n_cells = (self.win_size[0] // self.cell_size[0],
                  self.win_size[1] // self.cell_size[1])
        cell_hists = np.zeros((n_cells[1], n_cells[0], self.nbins))

        # Calculate contribution of each pixel
        for y in range(magnitude.shape[0]):
            for x in range(magnitude.shape[1]):
                mag = magnitude[y, x]
                ang = angle[y, x]

                # Calculate bin index
                bin_width = 180.0 / self.nbins
                bin_idx = int(ang / bin_width)
                next_bin = (bin_idx + 1) % self.nbins
                alpha = (ang - bin_idx * bin_width) / bin_width

                # Calculate cell index
                cell_x = x // self.cell_size[0]
                cell_y = y // self.cell_size[1]

                if cell_x < n_cells[0] and cell_y < n_cells[1]:
                    cell_hists[cell_y, cell_x, bin_idx] += mag * (1 - alpha)
                    cell_hists[cell_y, cell_x, next_bin] += mag * alpha

        return cell_hists

    def compute_block_features(self, cell_hists):
        """Compute block features and normalize"""
        n_cells = (self.win_size[0] // self.cell_size[0],
                  self.win_size[1] // self.cell_size[1])
        n_blocks = ((n_cells[0] - self.block_size[0] // self.cell_size[0]) //
                   (self.block_stride[0] // self.cell_size[0]) + 1,
                   (n_cells[1] - self.block_size[1] // self.cell_size[1]) //
                   (self.block_stride[1] // self.cell_size[1]) + 1)

        features = []
        for by in range(0, n_cells[1] - self.block_size[1] // self.cell_size[1] + 1,
                       self.block_stride[1] // self.cell_size[1]):
            for bx in range(0, n_cells[0] - self.block_size[0] // self.cell_size[0] + 1,
                          self.block_stride[0] // self.cell_size[0]):
                # Extract cell histograms in block
                block_features = cell_hists[by:by + self.block_size[1] // self.cell_size[1],
                                         bx:bx + self.block_size[0] // self.cell_size[0]].flatten()

                # L2-Norm normalization
                norm = np.sqrt(np.sum(block_features**2) + 1e-5)
                block_features = block_features / norm

                features.extend(block_features)

        return np.array(features)

    def compute(self, img):
        """Compute HOG features"""
        # Resize image
        img = cv2.resize(img, self.win_size)

        # Compute gradients
        magnitude, angle = self.compute_gradients(img)

        # Compute cell histogram
        cell_hists = self.compute_cell_histogram(magnitude, angle)

        # Compute block features and normalize
        features = self.compute_block_features(cell_hists)

        return features
```

## 4. Haar+AdaBoost Detection

### 4.1 Algorithm Principles

Haar features are like "light-dark contrast clues" sought by the detective, while AdaBoost is like a "decision committee" composed of multiple experts.

Haar feature calculation:
$$
f = \sum_{i \in white} I(i) - \sum_{i \in black} I(i)
$$

### 4.2 Implementation Example

```cpp
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

// Usage example
Mat detectFaces(const Mat& img) {
    Mat result = img.clone();
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    HaarExtractor haar;
    vector<Rect> faces = haar.detect(gray);

    for (const Rect& face : faces) {
        rectangle(result, face, Scalar(0, 255, 0), 2);
    }

    return result;
}
```

```python
def haar_adaboost_detection(img_path):
    """
    Haar+AdaBoost detection implementation
    Parameters:
        img_path: Input image path
    Returns:
        Detection result visualization
    """
    # Read image
    img = cv2.imread(img_path)
    result = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load pre-trained face detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw detection results
    for (x, y, w, h) in faces:
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return result
```

## 5. Non-Maximum Suppression

### 5.1 Algorithm Principles

Non-maximum suppression is like a detective organizing duplicate clues, keeping only the most significant findings.

NMS algorithm steps:
1. Sort by confidence
2. Calculate overlap
3. Suppress overlapping boxes

### 5.2 Implementation Example

```cpp
// Non-maximum suppression implementation
vector<int> nms(const vector<Rect>& boxes, const vector<float>& scores,
                float iou_threshold) {

    vector<int> indices(boxes.size());
    std::iota(indices.begin(), indices.end(), 0);  // Fill with 0, 1, 2, ...

    // Sort by score
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
```

```python
def compute_nms_manual(boxes, scores, iou_threshold=0.5):
    """
    Manual implementation of non-maximum suppression algorithm
    Parameters:
        boxes: List of bounding box coordinates, each box in [x1, y1, x2, y2] format
        scores: Confidence scores for each bounding box
        iou_threshold: IoU threshold, default 0.5
    Returns:
        keep: List of indices of kept bounding boxes
    """
    # Calculate areas of all boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    # Sort by score
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        # Calculate IoU
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        # Keep boxes with IoU less than threshold
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    return keep
```

## 6. Object Tracking

### 6.1 Basic Principles

Object tracking is like a detective continuously following important clues, considering:
1. Motion prediction
2. Feature matching
3. Trajectory smoothing
4. Occlusion handling

### 6.2 Implementation Example

```cpp
// Simple object tracker implementation
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

// Draw detection results
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
```

```python
def object_tracking(img_path, video_path=None):
    """
    Simple object tracking implementation
    Parameters:
        img_path: Input image path
        video_path: Video path (optional)
    Returns:
        Tracking result visualization
    """
    # Read image
    img = cv2.imread(img_path)
    result = img.copy()

    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define target color range (red as example)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])

    # Create mask
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find largest contour
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return result
```

## 7. Code Implementation and Optimization

### 7.1 Performance Optimization Tips

1. Use integral image to accelerate feature computation
2. GPU acceleration for large-scale computation
3. Multi-threaded parallel processing
4. Feature pyramid caching

### 7.2 Optimization Example

```cpp
// Optimize feature computation using integral image
class IntegralFeature {
private:
    Mat integralImg;
    Mat integralSqImg;

public:
    void compute(const Mat& img) {
        integral(img, integralImg, integralSqImg);
    }

    float getSum(const Rect& r) const {
        return integralImg.at<double>(r.y+r.height,r.x+r.width)
             + integralImg.at<double>(r.y,r.x)
             - integralImg.at<double>(r.y,r.x+r.width)
             - integralImg.at<double>(r.y+r.height,r.x);
    }

    float getVariance(const Rect& r) const {
        float area = r.width * r.height;
        float sum = getSum(r);
        float sqSum = getSqSum(r);
        return (sqSum - sum*sum/area) / area;
    }
};
```

## 8. Applications and Best Practices

### 8.1 Typical Applications

- 👤 Face detection
- 🚗 Vehicle detection
- 🚶 Pedestrian detection
- 📝 Text detection
- 🎯 Defect detection

### 8.2 Best Practices

1. Data Preparation
   - Sufficient training data
   - Data augmentation
   - Annotation quality control

2. Algorithm Selection
   - Choose appropriate algorithm based on scenario
   - Consider real-time requirements
   - Balance accuracy and speed

3. Deployment Optimization
   - Model compression
   - Computation acceleration
   - Memory optimization

## Summary

Object detection is like playing a "spot the difference" game in images, where we need to find specific targets in complex scenes! Through methods like sliding window, HOG+SVM, and Haar+AdaBoost, we can effectively locate and identify these targets. In practical applications, we need to choose appropriate methods based on specific scenarios, just like choosing different "magnifying glasses" to observe different targets.

Remember: A good object detection system is like an experienced "image detective" that can discover important targets from complex scenes! 🔍

## References

1. Viola P, Jones M. Rapid object detection using a boosted cascade of simple features[C]. CVPR, 2001
2. Dalal N, Triggs B. Histograms of oriented gradients for human detection[C]. CVPR, 2005
3. OpenCV Official Documentation: https://docs.opencv.org/
4. More Resources: [IP101 Project Homepage](https://github.com/GlimmerLab/IP101)