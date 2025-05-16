# 入门指南

## 1. 环境配置

### 1.1 系统要求
- 操作系统: Windows 10/11, Linux, macOS
- CPU: 支持AVX2指令集
- 内存: 8GB以上

### 1.2 依赖安装
```bash
# C++依赖
sudo apt-get install build-essential cmake libopencv-dev

# Python依赖
pip install -r requirements.txt
```

## 2. 编译安装

### 2.1 从源码编译
```bash
mkdir build && cd build
cmake ..
make -j8
```

### 2.2 安装Python包
```bash
pip install .
```

## 3. 快速开始

### 3.1 C++示例
```cpp
#include <image_processing.h>

int main() {
    // 读取图像
    Mat image = cv::imread("input.jpg");

    // 创建处理器
    ImageProcessor processor;

    // 应用高斯滤波
    Mat result = processor.gaussianBlur(image, 5, 1.0);

    // 保存结果
    cv::imwrite("output.jpg", result);
    return 0;
}
```

### 3.2 Python示例
```python
from image_processing import ImageProcessor
import cv2

# 读取图像
image = cv2.imread("input.jpg")

# 创建处理器
processor = ImageProcessor()

# 应用高斯滤波
result = processor.gaussian_blur(image, ksize=5, sigma=1.0)

# 保存结果
cv2.imwrite("output.jpg", result)
```

### 3.3 命令行工具
```bash
# 高斯滤波
image_processing blur input.jpg output.jpg --ksize 5 --sigma 1.0

# 边缘检测
image_processing edge input.jpg output.jpg
```

## 4. 项目结构

```
project/
├── cpp/          # C++源码
├── python/       # Python源码
├── docs/         # 文档
├── examples/     # 示例代码
└── tests/        # 测试代码
```

## 5. 下一步

1. 阅读[算法文档](../algorithms/basic/README.md)
2. 查看[优化指南](../optimization/README.md)
3. 运行[示例程序](../../examples/)