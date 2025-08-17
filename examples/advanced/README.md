# IP101 高级算法示例程序

本目录包含了 IP101 项目中所有高级图像处理算法的完整使用示例。

## 📁 示例程序列表

### 🔧 高级滤波算法
- **guided_filter_test.cpp** - 导向滤波算法示例
  - 支持多种导向滤波变体
  - 性能对比测试
  - 参数效果分析

- **side_window_filter_test.cpp** - 侧窗滤波算法示例
  - 边缘保持滤波
  - 与OpenCV双边滤波对比
  - 不同参数效果测试

- **homomorphic_filter_test.cpp** - 同态滤波算法示例
  - 光照校正
  - 频率域分析
  - 不同滤波参数测试

### 🎨 图像矫正算法
- **automatic_white_balance_test.cpp** - 自动白平衡算法示例
  - 色温校正
  - 不同光照条件测试
  - 直方图分析

- **gamma_correction_test.cpp** - 伽马校正算法示例
  - 显示校正
  - 不同伽马值效果
  - 直方图对比分析

### 🌫️ 图像去雾算法
- **dark_channel_test.cpp** - 暗通道去雾算法示例
  - 雾霾模拟与去除
  - 透射率图分析
  - 参数优化测试

### ✨ 图像增强算法
- **retinex_msrcr_test.cpp** - Retinex MSRCR算法示例
  - 低光照增强
  - 多尺度细节增强
  - 参数效果分析

### 🎭 图像特效算法
- **cartoon_effect_test.cpp** - 卡通效果算法示例
  - 不同风格效果
  - 参数调节测试
  - 边缘保持效果

### 🔍 特殊检测算法
- **rectangle_detection_test.cpp** - 矩形检测算法示例
  - 多场景测试
  - 检测结果分析
  - 参数敏感性测试

## 🚀 使用方法

### 编译所有示例
```bash
# 在项目根目录下
mkdir build && cd build
cmake ..
make advanced_examples
```

### 运行单个示例
```bash
# 基本用法
./guided_filter_test.exe <image_path>

# 示例
./guided_filter_test.exe assets/imori.jpg
```

### 运行所有示例测试
```bash
# Windows
quick_test_advanced.bat

# Linux/macOS
./quick_test_advanced.sh
```

## 📊 测试功能

每个示例程序都包含以下测试功能：

### 1. 性能测试
- 算法执行时间测量
- 与OpenCV对应算法性能对比
- 多次运行取平均值

### 2. 参数效果测试
- 不同参数组合的效果
- 参数对性能的影响
- 最佳参数推荐

### 3. 可视化结果
- 多参数结果对比显示
- 原图与处理结果对比
- 交互式结果查看

### 4. 特殊场景测试
- 不同光照条件
- 噪声和模糊影响
- 极端参数测试

## 📁 输出文件

每个示例程序会生成以下类型的输出文件：

- **性能测试结果** - 控制台输出
- **参数效果图像** - 不同参数的处理结果
- **对比分析图像** - 原图与处理结果对比
- **直方图分析** - 图像统计信息可视化
- **检测结果标注** - 检测算法的可视化结果

## 🔧 自定义测试

### 修改测试参数
每个示例程序都允许通过命令行参数或代码修改来调整测试参数：

```cpp
// 示例：修改导向滤波参数
ip101::advanced::guided_filter(src, dst,
    radius,    // 滤波半径
    eps        // 正则化参数
);
```

### 添加新的测试场景
可以在示例程序中添加新的测试函数：

```cpp
void test_custom_scenario(const cv::Mat& src) {
    cv::Mat dst;

    // 自定义处理
    ip101::advanced::your_algorithm(src, dst, custom_params);

    // 保存结果
    cv::imwrite("custom_result.jpg", dst);
}
```

## 📈 性能分析

### 性能指标
- **执行时间** - 毫秒级精度
- **内存使用** - 算法内存占用
- **质量评估** - 处理效果量化

### 优化建议
- 根据图像尺寸选择合适的参数
- 考虑实时性要求的参数调节
- 平衡质量和性能的最佳实践

## 🐛 故障排除

### 常见问题
1. **编译错误** - 检查OpenCV版本兼容性
2. **运行时错误** - 确认输入图像路径正确
3. **性能问题** - 调整算法参数或图像尺寸

### 调试技巧
- 使用较小的测试图像进行快速验证
- 检查控制台输出的错误信息
- 对比不同参数的处理结果

## 📚 相关文档

- [算法原理文档](../docs/algorithms/advanced/)
- [API参考文档](../docs/api/)
- [性能优化指南](../docs/optimization/)

---

*这些示例程序展示了IP101高级算法的完整功能和使用方法，为开发者提供了实用的参考实现。*
