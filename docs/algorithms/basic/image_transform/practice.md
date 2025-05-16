# 图像变换实战练习

## 1. 图像旋转与缩放

### 问题描述
给定一张图片,实现以下功能:
1. 将图片旋转45度
2. 将图片缩小到原来的一半
3. 将图片放大到原来的2倍

### 实现步骤
1. 读取输入图片
2. 计算旋转矩阵
3. 应用旋转变换
4. 计算缩放矩阵
5. 应用缩放变换
6. 保存结果图片

### 代码实现
```python
import cv2
import numpy as np

def rotate_image(image, angle):
    # 获取图像尺寸
    (h, w) = image.shape[:2]
    # 计算旋转中心
    center = (w // 2, h // 2)
    # 计算旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # 应用旋转
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def scale_image(image, scale):
    # 计算新的尺寸
    new_width = int(image.shape[1] * scale)
    new_height = int(image.shape[0] * scale)
    # 应用缩放
    scaled = cv2.resize(image, (new_width, new_height))
    return scaled

# 读取图片
image = cv2.imread('input.jpg')

# 旋转45度
rotated = rotate_image(image, 45)
cv2.imwrite('rotated.jpg', rotated)

# 缩小到一半
scaled_down = scale_image(image, 0.5)
cv2.imwrite('scaled_down.jpg', scaled_down)

# 放大到2倍
scaled_up = scale_image(image, 2.0)
cv2.imwrite('scaled_up.jpg', scaled_up)
```

## 2. 图像透视变换

### 问题描述
给定一张图片和一个四边形区域,实现以下功能:
1. 将四边形区域变换为矩形
2. 将矩形区域变换为任意四边形

### 实现步骤
1. 读取输入图片
2. 定义源点和目标点
3. 计算透视变换矩阵
4. 应用透视变换
5. 保存结果图片

### 代码实现
```python
import cv2
import numpy as np

def perspective_transform(image, src_points, dst_points):
    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    # 应用透视变换
    transformed = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))
    return transformed

# 读取图片
image = cv2.imread('input.jpg')

# 定义源点(四边形)
src_points = np.float32([[100, 100], [400, 100], [400, 400], [100, 400]])

# 定义目标点(矩形)
dst_points = np.float32([[0, 0], [300, 0], [300, 300], [0, 300]])

# 将四边形变换为矩形
rectified = perspective_transform(image, src_points, dst_points)
cv2.imwrite('rectified.jpg', rectified)

# 定义新的目标点(任意四边形)
new_dst_points = np.float32([[50, 50], [350, 50], [400, 350], [0, 350]])

# 将矩形变换为任意四边形
transformed = perspective_transform(rectified, dst_points, new_dst_points)
cv2.imwrite('transformed.jpg', transformed)
```

## 3. 图像镜像与平移

### 问题描述
给定一张图片,实现以下功能:
1. 水平镜像
2. 垂直镜像
3. 向右平移100像素
4. 向下平移100像素

### 实现步骤
1. 读取输入图片
2. 计算镜像矩阵
3. 应用镜像变换
4. 计算平移矩阵
5. 应用平移变换
6. 保存结果图片

### 代码实现
```python
import cv2
import numpy as np

def mirror_image(image, direction='horizontal'):
    if direction == 'horizontal':
        # 水平镜像
        mirrored = cv2.flip(image, 1)
    else:
        # 垂直镜像
        mirrored = cv2.flip(image, 0)
    return mirrored

def translate_image(image, tx, ty):
    # 创建平移矩阵
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    # 应用平移
    translated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return translated

# 读取图片
image = cv2.imread('input.jpg')

# 水平镜像
h_mirrored = mirror_image(image, 'horizontal')
cv2.imwrite('h_mirrored.jpg', h_mirrored)

# 垂直镜像
v_mirrored = mirror_image(image, 'vertical')
cv2.imwrite('v_mirrored.jpg', v_mirrored)

# 向右平移100像素
right_translated = translate_image(image, 100, 0)
cv2.imwrite('right_translated.jpg', right_translated)

# 向下平移100像素
down_translated = translate_image(image, 0, 100)
cv2.imwrite('down_translated.jpg', down_translated)
```

## 4. 综合应用：图像校正

### 问题描述
给定一张倾斜的文档图片,实现以下功能:
1. 检测文档边缘
2. 计算文档的四个角点
3. 将文档校正为矩形
4. 调整文档大小为标准A4尺寸

### 实现步骤
1. 读取输入图片
2. 预处理(灰度化、二值化等)
3. 边缘检测
4. 轮廓检测
5. 角点检测
6. 计算透视变换矩阵
7. 应用透视变换
8. 调整大小
9. 保存结果图片

### 代码实现
```python
import cv2
import numpy as np

def document_correction(image):
    # 灰度化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 高斯模糊
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 边缘检测
    edges = cv2.Canny(blurred, 50, 150)

    # 轮廓检测
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 找到最大的轮廓(假设是文档)
    max_contour = max(contours, key=cv2.contourArea)

    # 计算轮廓的近似多边形
    epsilon = 0.02 * cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, epsilon, True)

    # 确保找到4个点
    if len(approx) == 4:
        # 排序角点(左上、右上、右下、左下)
        points = np.float32(approx.reshape(4, 2))
        rect = np.zeros((4, 2), dtype=np.float32)

        # 计算中心点
        center = np.mean(points, axis=0)

        # 根据与中心点的位置关系排序
        for point in points:
            if point[0] < center[0] and point[1] < center[1]:
                rect[0] = point  # 左上
            elif point[0] > center[0] and point[1] < center[1]:
                rect[1] = point  # 右上
            elif point[0] > center[0] and point[1] > center[1]:
                rect[2] = point  # 右下
            else:
                rect[3] = point  # 左下

        # 计算目标点(A4尺寸)
        width = 210 * 4  # A4宽度(毫米) * 4像素/毫米
        height = 297 * 4  # A4高度(毫米) * 4像素/毫米
        dst = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(rect, dst)

        # 应用透视变换
        corrected = cv2.warpPerspective(image, M, (width, height))

        return corrected
    else:
        print("未找到合适的文档轮廓")
        return None

# 读取图片
image = cv2.imread('document.jpg')

# 校正文档
corrected = document_correction(image)
if corrected is not None:
    cv2.imwrite('corrected_document.jpg', corrected)
```

## 5. 进阶练习：图像拼接

### 问题描述
给定两张有重叠区域的图片,实现以下功能:
1. 检测两张图片的特征点
2. 匹配特征点
3. 计算单应性矩阵
4. 将第二张图片变换到第一张图片的坐标系
5. 拼接两张图片

### 实现步骤
1. 读取两张输入图片
2. 特征点检测
3. 特征点匹配
4. 计算单应性矩阵
5. 应用透视变换
6. 图像拼接
7. 保存结果图片

### 代码实现
```python
import cv2
import numpy as np

def stitch_images(image1, image2):
    # 创建SIFT特征检测器
    sift = cv2.SIFT_create()

    # 检测特征点和描述子
    kp1, des1 = sift.detectAndCompute(image1, None)
    kp2, des2 = sift.detectAndCompute(image2, None)

    # 创建FLANN匹配器
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # 特征点匹配
    matches = flann.knnMatch(des1, des2, k=2)

    # 筛选好的匹配点
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # 获取匹配点的坐标
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 计算单应性矩阵
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # 计算拼接后图片的大小
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    corners2_transformed = cv2.perspectiveTransform(corners2, H)

    # 计算拼接后图片的尺寸
    corners = np.concatenate((corners1, corners2_transformed), axis=0)
    [xmin, ymin] = np.int32(corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(corners.max(axis=0).ravel() + 0.5)
    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])

    # 应用变换
    result = cv2.warpPerspective(image1, Ht.dot(H), (xmax-xmin, ymax-ymin))
    result[t[1]:h2+t[1], t[0]:w2+t[0]] = image2

    return result

# 读取图片
image1 = cv2.imread('image1.jpg')
image2 = cv2.imread('image2.jpg')

# 拼接图片
stitched = stitch_images(image1, image2)
cv2.imwrite('stitched.jpg', stitched)
```

## 6. 图像拼接魔法 🧩

### 问题描述
创建一个图像拼接程序，实现以下功能：
1. 全景图像拼接
2. 多视角图像合成
3. 实时视频拼接

### 实现步骤
1. 读取输入图像
2. 特征点检测与匹配
3. 计算变换矩阵
4. 图像对齐和融合
5. 保存结果图像

### 代码实现
```python
def create_image_mosaic(images, rows, cols):
    """
    创建图像拼接马赛克

    参数:
        images: 输入图像列表
        rows: 行数
        cols: 列数
    """
    # 1. 调整所有图像大小
    # 2. 应用不同的变换
    # 3. 拼接到一起
    pass

# 示例任务：
# 1. 创建2x2的图像网格
# 2. 对每个图像应用不同的变换
# 3. 无缝拼接
```

## 7. 文档扫描器 📄

### 问题描述
实现一个智能文档扫描器，具备以下功能：
1. 智能边缘检测
2. 自动透视校正
3. 文档增强处理

### 实现步骤
1. 图像预处理
2. 边缘检测
3. 角点检测
4. 透视变换
5. 图像增强

### 代码实现
```python
def document_scanner(image_path):
    """
    文档扫描器

    步骤：
    1. 检测文档边缘
    2. 应用透视变换
    3. 增强对比度
    """
    # 实现代码
    pass

# 挑战：
# 1. 自动检测文档边缘
# 2. 处理不同光照条件
# 3. 优化扫描质量
```

## 8. 图像变换艺术 🎨

### 问题描述
创建一个艺术效果生成器，实现以下特效：
1. 万花筒效果
2. 波浪变形
3. 旋涡特效

### 实现步骤
1. 基础变换实现
2. 特效参数设计
3. 图像处理流程
4. 效果优化

### 代码实现
```python
def create_art_effect(image, effect_type):
    """
    创建艺术效果

    效果类型：
    - 万花筒
    - 波浪
    - 旋涡
    """
    if effect_type == 'kaleidoscope':
        # 使用旋转和镜像创建万花筒效果
        pass
    elif effect_type == 'wave':
        # 使用正弦波扭曲创建波浪效果
        pass
    elif effect_type == 'swirl':
        # 使用极坐标变换创建旋涡效果
        pass

# 创意挑战：
# 1. 设计新的艺术效果
# 2. 添加交互控制
# 3. 优化渲染性能
```

## 9. 实时变换应用 📱

### 问题描述
开发一个实时图像变换应用，包含：
1. 实时镜像
2. 动态旋转
3. 缩放预览

### 实现步骤
1. 视频流获取
2. 实时处理
3. 效果展示
4. 性能优化

### 代码实现
```python
def real_time_transform(transform_type='rotate'):
    """
    实时图像变换演示
    """
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 根据用户输入选择变换
        if transform_type == 'rotate':
            # 实时旋转
            angle = time.time() * 30  # 每秒旋转30度
            frame = rotate_image(frame, angle)
        elif transform_type == 'wave':
            # 实时波浪效果
            frame = apply_wave_effect(frame, time.time())

        cv2.imshow('Real-time Transform', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 进阶挑战：
# 1. 添加更多实时效果
# 2. 优化实时性能
# 3. 添加用户交互
```

## 10. 图像校正大师 📐

### 问题描述
实现一个综合图像校正工具，包含：
1. 智能倾斜校正
2. 畸变矫正
3. 透视校正

### 实现步骤
1. 图像分析
2. 参数估计
3. 校正变换
4. 结果优化

### 代码实现
```python
class ImageCorrectionMaster:
    """
    图像校正大师
    """
    def __init__(self):
        self.image = None
        self.correction_params = {}

    def load_image(self, path):
        """加载图像"""
        self.image = cv2.imread(path)

    def detect_skew(self):
        """检测倾斜角度"""
        # 实现倾斜检测算法
        pass

    def correct_distortion(self):
        """校正畸变"""
        # 实现畸变校正
        pass

    def correct_perspective(self):
        """校正透视变形"""
        # 实现透视校正
        pass

    def auto_correct(self):
        """自动校正"""
        # 1. 检测问题
        # 2. 估计参数
        # 3. 应用校正
        pass

# 高级功能：
# 1. 自动检测问题类型
# 2. 智能参数估计
# 3. 批量处理
# 4. 实时预览
```

## 11. 图像变换工具箱 🛠️

### 问题描述
创建一个完整的图像变换工具箱，具备：
1. 基础变换功能
2. 历史记录管理
3. 批处理能力
4. 实时预览

### 实现步骤
1. 核心功能实现
2. 历史记录系统
3. 批处理模块
4. 用户界面

### 代码实现
```python
class ImageTransformToolbox:
    """
    图像变换工具箱
    """
    def __init__(self):
        self.history = []  # 变换历史
        self.image = None

    def load_image(self, path):
        """加载图像"""
        self.image = cv2.imread(path)
        self.history.append(('load', path))

    def apply_transform(self, transform_type, **params):
        """应用变换"""
        if transform_type == 'rotate':
            self.image = rotate_image(self.image, **params)
        elif transform_type == 'scale':
            self.image = scale_image(self.image, **params)
        # ... 其他变换

        self.history.append((transform_type, params))

    def undo(self):
        """撤销上一步变换"""
        if len(self.history) > 1:
            self.history.pop()
            self._replay_history()

    def _replay_history(self):
        """重放变换历史"""
        original_history = self.history.copy()
        self.history = []
        self.load_image(original_history[0][1])

        for transform_type, params in original_history[1:]:
            self.apply_transform(transform_type, **params)

# 项目要求：
# 1. 实现所有基本变换
# 2. 添加变换历史记录
# 3. 支持撤销/重做
# 4. 优化性能
# 5. 添加批处理功能
```

## 性能优化挑战 🚀

### 问题描述
优化图像变换的性能，关注：
1. 处理速度
2. 内存使用
3. 代码效率

### 实现步骤
1. 性能分析
2. 算法优化
3. 代码重构
4. 测试验证

### 代码实现
```python
def benchmark_transforms():
    """
    性能测试不同的实现方法
    """
    image = cv2.imread('test.jpg')
    results = {}

    # 1. 基础实现
    start = time.time()
    basic_result = basic_transform(image)
    results['basic'] = time.time() - start

    # 2. NumPy优化
    start = time.time()
    numpy_result = numpy_transform(image)
    results['numpy'] = time.time() - start

    # 3. SIMD优化
    start = time.time()
    simd_result = simd_transform(image)
    results['simd'] = time.time() - start

    return results

# 优化目标：
# 1. 提高处理速度
# 2. 减少内存使用
# 3. 保持图像质量
```

> 💡 提示：
- 从简单开始，逐步添加功能
- 注意代码的可维护性
- 考虑实际应用场景
- 持续优化性能

祝你编码愉快！🎉