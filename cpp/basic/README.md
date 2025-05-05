# 基础图像处理算法优化实现

[English](README_EN.md) | 简体中文

本目录包含了基础图像处理算法的C++实现，所有实现都经过了深度优化，以接近或达到OpenCV的性能水平。

## 🚀 编译要求

- C++17 或更高版本
- OpenCV 4.x
- 支持 AVX2/SSE4.2 的CPU
- OpenMP 支持
- CMake 3.10+

## 📦 编译选项

所有文件都使用以下基本编译选项：
```bash
g++ -std=c++17 -O3 -march=native -fopenmp -mavx2 -mfma -msse4.2 [source_file] -o [output_file] `pkg-config --cflags --libs opencv4`
```

### 各文件特定编译选项

1. **filtering.cpp** (图像滤波)
```bash
g++ -std=c++17 -O3 -march=native -fopenmp -mavx2 -mfma -msse4.2 filtering.cpp -o filtering `pkg-config --cflags --libs opencv4`
```
优化重点：
- SIMD (AVX2/SSE4) 向量化
- OpenMP 多线程并行
- 缓存优化和内存对齐
- 分块处理
- 边界预处理

2. **edge_detection.cpp** (边缘检测)
```bash
g++ -std=c++17 -O3 -march=native -fopenmp -mavx2 -mfma -msse4.2 edge_detection.cpp -o edge_detection `pkg-config --cflags --libs opencv4`
```
优化重点：
- SIMD 向量化处理梯度计算
- 分离卷积优化
- 并行处理
- 查找表优化
- 定点数计算

3. **color_operations.cpp** (颜色操作)
```bash
g++ -std=c++17 -O3 -march=native -fopenmp -mavx2 -mfma -msse4.2 color_operations.cpp -o color_operations `pkg-config --cflags --libs opencv4`
```
优化重点：
- SIMD 颜色转换
- 并行处理
- 查找表优化
- 内存访问优化
- 分块处理

## 🔧 优化策略说明

### 1. SIMD 优化
- 使用 AVX2 进行大批量数据处理
- 使用 SSE4.2 处理特定算法
- 合理使用向量指令集
- 注意内存对齐

### 2. 内存优化
- 缓存行对齐
- 减少内存拷贝
- 优化内存访问模式
- 使用分块处理

### 3. 并行优化
- OpenMP 动态调度
- 自适应线程数
- 任务分块
- 负载均衡

### 4. 算法优化
- 查找表预计算
- 定点数运算
- 分离卷积
- 边界预处理

### 5. 编译优化
- 使用最高优化级别
- 启用自动向量化
- 启用链接时优化
- 使用CPU特定指令集

## 📊 性能对比

每个实现都包含了与OpenCV的性能对比测试。测试结果显示：
- 均值滤波：性能达到OpenCV的80-90%
- 中值滤波：性能达到OpenCV的70-80%
- 高斯滤波：性能达到OpenCV的75-85%
- 边缘检测：性能达到OpenCV的85-95%
- 颜色转换：性能达到OpenCV的90-100%

## 🔍 使用示例

每个文件都包含了完整的示例代码和性能测试函数。运行示例：

```bash
# 编译
g++ -O3 -fopenmp -mavx2 -mfma -msse4.2 filtering.cpp -o filtering `pkg-config --cflags --libs opencv4`

# 运行
./filtering
```

## 📝 注意事项

1. 确保您的CPU支持相应的指令集
2. 根据实际硬件调整线程数和分块大小
3. 大图像处理时注意内存使用
4. 根据具体应用场景选择合适的优化策略

## 🔄 更新日志

- 2024-01: 初始版本，完成基础优化
- 2024-01: 添加SIMD优化
- 2024-01: 添加性能测试框架
- 2024-01: 完善文档说明