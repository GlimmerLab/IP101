# IP101 - 图像处理基础100问

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-blue.svg)](https://github.com/GlimmerLab/IP101)
[![Language](https://img.shields.io/badge/language-C%2B%2B%20%7C%20Python-orange.svg)](https://github.com/GlimmerLab/IP101)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green.svg)](https://opencv.org/)
[![CMake](https://img.shields.io/badge/CMake-3.10+-red.svg)](https://cmake.org/)

[English](README_EN.md) | 简体中文

IP101 是一个专注于图像处理基础知识、操作、应用和优化的系列教程。本系列旨在通过100个精心设计的问题，帮助读者全面掌握图像处理的核心概念和实践技能。

## 🚀 快速开始

```bash
# 克隆项目
git clone https://github.com/GlimmerLab/IP101.git
cd IP101

# Python用户：直接运行
python python/basic/color_operations.py 1

# C++用户：构建项目
mkdir build && cd build
cmake ..
cmake --build . --config Release
./examples/basic/color_operations_test
```

## 📋 目录
- [项目结构](#项目结构)
- [内容特点](#内容特点)
- [基础问题分类](#基础问题分类)
- [进阶算法列表](#进阶算法列表)
- [使用说明](#使用说明)
- [贡献指南](#贡献指南)

## 项目结构

```
IP101/
├── include/            # 头文件目录
│   ├── basic/         # 基础算法头文件
│   └── advanced/      # 进阶算法头文件
│       ├── correction/    # 图像校正算法
│       ├── defogging/     # 图像去雾算法
│       ├── detection/     # 特殊检测算法
│       ├── effects/       # 图像特效算法
│       ├── enhancement/   # 图像增强算法
│       └── filtering/     # 高级滤波算法
├── cpp/                # C++实现
│   ├── basic/          # 基础问题代码
│   └── advanced/       # 进阶算法代码
│       ├── image_correction/    # 图像校正
│       ├── image_defogging/     # 图像去雾
│       ├── image_effects/       # 图像特效
│       ├── image_enhancement/   # 图像增强
│       ├── advanced_filtering/  # 高级滤波
│       └── special_detection/   # 特殊检测
├── python/             # Python实现
│   ├── basic/          # 基础问题代码
│   ├── advanced/       # 进阶算法代码
│   ├── image_processing/   # 图像处理工具
│   └── tests/          # 测试代码
├── examples/           # 示例代码
│   ├── basic/          # 基础问题示例
│   └── advanced/       # 进阶算法示例
├── docs/               # 文档目录
│   ├── algorithms/     # 算法文档
│   ├── tutorials/      # 教程文档
│   └── optimization/   # 优化技术文档
├── gui/                # GUI界面
├── tests/              # 测试代码
├── utils/              # 工具函数
├── third_party/        # 第三方依赖
│   ├── glfw/           # GLFW库
│   └── imgui/          # ImGui库
├── cmake/              # CMake配置
└── assets/             # 资源文件
```

## ✨ 项目特色

### 🎯 教学导向
- **手动实现**：所有算法均为手动实现，不使用OpenCV内置函数，帮助理解算法原理
- **循序渐进**：从基础到进阶的完整学习路径
- **理论与实践结合**：每个算法都配有详细的数学原理说明

### 🚀 技术优势
- **高性能**：C++实现支持SIMD优化和多线程加速
- **跨平台**：支持Windows、Linux、macOS
- **易扩展**：模块化设计，便于添加新算法

### 📚 内容丰富
- **100个基础问题**：涵盖图像处理的各个领域
- **30+进阶算法**：包含最新的研究算法
- **双语支持**：中英文文档和代码注释

## 支持的语言

| 语言 | 状态 | 说明 |
|------|------|------|
| Python | ✅ | 完整支持，包含所有100个问题的解答 |
| C++ | ✅ | 完整支持，包含所有100个问题的解答 |
| MATLAB | ❌ | 暂不支持 |

## 基础问题分类

### 1. 颜色操作 (color_operations.py)
| 问题编号 | 问题名称 | 难度等级 | 代码索引 |
|----------|----------|----------|----------|
| Q1 | 通道替换 | ⭐ | [Python](python/basic/color_operations.py) / [C++](cpp/basic/color_operations.cpp) |
| Q2 | 灰度化 | ⭐ | [Python](python/basic/color_operations.py) / [C++](cpp/basic/color_operations.cpp) |
| Q3 | 二值化 | ⭐ | [Python](python/basic/color_operations.py) / [C++](cpp/basic/color_operations.cpp) |
| Q4 | 大津算法 | ⭐⭐ | [Python](python/basic/color_operations.py) / [C++](cpp/basic/color_operations.cpp) |
| Q5 | HSV变换 | ⭐⭐ | [Python](python/basic/color_operations.py) / [C++](cpp/basic/color_operations.cpp) |

### 2. 图像滤波 (filtering.py)
| 问题编号 | 问题名称 | 难度等级 | 代码索引 |
|----------|----------|----------|----------|
| Q6 | 均值滤波 | ⭐ | [Python](python/basic/filtering.py) / [C++](cpp/basic/filtering.cpp) |
| Q7 | 中值滤波 | ⭐⭐ | [Python](python/basic/filtering.py) / [C++](cpp/basic/filtering.cpp) |
| Q8 | 高斯滤波 | ⭐⭐ | [Python](python/basic/filtering.py) / [C++](cpp/basic/filtering.cpp) |
| Q9 | 均值池化 | ⭐ | [Python](python/basic/filtering.py) / [C++](cpp/basic/filtering.cpp) |
| Q10 | Max池化 | ⭐ | [Python](python/basic/filtering.py) / [C++](cpp/basic/filtering.cpp) |

### 3. 边缘检测 (edge_detection.py)
| 问题编号 | 问题名称 | 难度等级 | 代码索引 |
|----------|----------|----------|----------|
| Q11 | 微分滤波 | ⭐⭐ | [Python](python/basic/edge_detection.py) / [C++](cpp/basic/edge_detection.cpp) |
| Q12 | Sobel滤波 | ⭐⭐ | [Python](python/basic/edge_detection.py) / [C++](cpp/basic/edge_detection.cpp) |
| Q13 | Prewitt滤波 | ⭐⭐ | [Python](python/basic/edge_detection.py) / [C++](cpp/basic/edge_detection.cpp) |
| Q14 | Laplacian滤波 | ⭐⭐ | [Python](python/basic/edge_detection.py) / [C++](cpp/basic/edge_detection.cpp) |
| Q15 | 浮雕效果 | ⭐⭐ | [Python](python/basic/edge_detection.py) / [C++](cpp/basic/edge_detection.cpp) |
| Q16 | 边缘检测 | ⭐⭐ | [Python](python/basic/edge_detection.py) / [C++](cpp/basic/edge_detection.cpp) |

### 4. 图像变换 (image_transform.py)
| 问题编号 | 问题名称 | 难度等级 | 代码索引 |
|----------|----------|----------|----------|
| Q17 | 仿射变换 | ⭐⭐ | [Python](python/basic/image_transform.py) / [C++](cpp/basic/image_transform.cpp) |
| Q18 | 透视变换 | ⭐⭐ | [Python](python/basic/image_transform.py) / [C++](cpp/basic/image_transform.cpp) |
| Q19 | 旋转 | ⭐⭐ | [Python](python/basic/image_transform.py) / [C++](cpp/basic/image_transform.cpp) |
| Q20 | 缩放 | ⭐⭐ | [Python](python/basic/image_transform.py) / [C++](cpp/basic/image_transform.cpp) |
| Q21 | 平移 | ⭐ | [Python](python/basic/image_transform.py) / [C++](cpp/basic/image_transform.cpp) |
| Q22 | 镜像 | ⭐ | [Python](python/basic/image_transform.py) / [C++](cpp/basic/image_transform.cpp) |

### 5. 图像增强 (image_enhancement.py)
| 问题编号 | 问题名称 | 难度等级 | 代码索引 |
|----------|----------|----------|----------|
| Q23 | 直方图均衡化 | ⭐⭐ | [Python](python/basic/image_enhancement.py) / [C++](cpp/basic/image_enhancement.cpp) |
| Q24 | 伽马变换 | ⭐⭐ | [Python](python/basic/image_enhancement.py) / [C++](cpp/basic/image_enhancement.cpp) |
| Q25 | 对比度拉伸 | ⭐⭐ | [Python](python/basic/image_enhancement.py) / [C++](cpp/basic/image_enhancement.cpp) |
| Q26 | 亮度调整 | ⭐ | [Python](python/basic/image_enhancement.py) / [C++](cpp/basic/image_enhancement.cpp) |
| Q27 | 饱和度调整 | ⭐⭐ | [Python](python/basic/image_enhancement.py) / [C++](cpp/basic/image_enhancement.cpp) |

### 6. 特征提取 (feature_extraction.py)
| 问题编号 | 问题名称 | 难度等级 | 代码索引 |
|----------|----------|----------|----------|
| Q28 | Harris角点检测 | ⭐⭐⭐ | [Python](python/basic/feature_extraction.py) / [C++](cpp/basic/feature_extraction.cpp) |
| Q29 | SIFT特征 | ⭐⭐⭐ | [Python](python/basic/feature_extraction.py) / [C++](cpp/basic/feature_extraction.cpp) |
| Q30 | SURF特征 | ⭐⭐⭐ | [Python](python/basic/feature_extraction.py) / [C++](cpp/basic/feature_extraction.cpp) |
| Q31 | ORB特征 | ⭐⭐⭐ | [Python](python/basic/feature_extraction.py) / [C++](cpp/basic/feature_extraction.cpp) |
| Q32 | 特征匹配 | ⭐⭐⭐ | [Python](python/basic/feature_extraction.py) / [C++](cpp/basic/feature_extraction.cpp) |

### 7. 图像分割 (image_segmentation.py)
| 问题编号 | 问题名称 | 难度等级 | 代码索引 |
|----------|----------|----------|----------|
| Q33 | 阈值分割 | ⭐⭐ | [Python](python/basic/image_segmentation.py) / [C++](cpp/basic/image_segmentation.cpp) |
| Q34 | K均值分割 | ⭐⭐⭐ | [Python](python/basic/image_segmentation.py) / [C++](cpp/basic/image_segmentation.cpp) |
| Q35 | 区域生长 | ⭐⭐⭐ | [Python](python/basic/image_segmentation.py) / [C++](cpp/basic/image_segmentation.cpp) |
| Q36 | 分水岭分割 | ⭐⭐⭐ | [Python](python/basic/image_segmentation.py) / [C++](cpp/basic/image_segmentation.cpp) |
| Q37 | 图割分割 | ⭐⭐⭐ | [Python](python/basic/image_segmentation.py) / [C++](cpp/basic/image_segmentation.cpp) |

### 8. 形态学处理 (morphology.py)
| 问题编号 | 问题名称 | 难度等级 | 代码索引 |
|----------|----------|----------|----------|
| Q38 | 膨胀操作 | ⭐⭐ | [Python](python/basic/morphology.py) / [C++](cpp/basic/morphology.cpp) |
| Q39 | 腐蚀操作 | ⭐⭐ | [Python](python/basic/morphology.py) / [C++](cpp/basic/morphology.cpp) |
| Q40 | 开运算 | ⭐⭐ | [Python](python/basic/morphology.py) / [C++](cpp/basic/morphology.cpp) |
| Q41 | 闭运算 | ⭐⭐ | [Python](python/basic/morphology.py) / [C++](cpp/basic/morphology.cpp) |
| Q42 | 形态学梯度 | ⭐⭐⭐ | [Python](python/basic/morphology.py) / [C++](cpp/basic/morphology.cpp) |

### 9. 频域处理 (frequency_domain.py)
| 问题编号 | 问题名称 | 难度等级 | 代码索引 |
|----------|----------|----------|----------|
| Q43 | 傅里叶变换 | ⭐⭐⭐ | [Python](python/basic/frequency_domain.py) / [C++](cpp/basic/frequency_domain.cpp) |
| Q44 | 频域滤波 | ⭐⭐⭐ | [Python](python/basic/frequency_domain.py) / [C++](cpp/basic/frequency_domain.cpp) |
| Q45 | DCT变换 | ⭐⭐⭐ | [Python](python/basic/frequency_domain.py) / [C++](cpp/basic/frequency_domain.cpp) |
| Q46 | 小波变换 | ⭐⭐⭐ | [Python](python/basic/frequency_domain.py) / [C++](cpp/basic/frequency_domain.cpp) |

### 10. 图像压缩 (image_compression.py)
| 问题编号 | 问题名称 | 难度等级 | 代码索引 |
|----------|----------|----------|----------|
| Q47 | 无损压缩 | ⭐⭐⭐ | [Python](python/basic/image_compression.py) / [C++](cpp/basic/image_compression.cpp) |
| Q48 | JPEG压缩 | ⭐⭐⭐ | [Python](python/basic/image_compression.py) / [C++](cpp/basic/image_compression.cpp) |
| Q49 | 分形压缩 | ⭐⭐⭐ | [Python](python/basic/image_compression.py) / [C++](cpp/basic/image_compression.cpp) |
| Q50 | 小波压缩 | ⭐⭐⭐ | [Python](python/basic/image_compression.py) / [C++](cpp/basic/image_compression.cpp) |

### 11. 图像特征 (image_features.py)
| 问题编号 | 问题名称 | 难度等级 | 代码索引 |
|----------|----------|----------|----------|
| Q51 | HOG特征提取 | ⭐⭐⭐ | [Python](python/basic/image_features.py) / [C++](cpp/basic/image_features.cpp) |
| Q52 | LBP特征提取 | ⭐⭐⭐ | [Python](python/basic/image_features.py) / [C++](cpp/basic/image_features.cpp) |
| Q53 | Haar特征提取 | ⭐⭐⭐ | [Python](python/basic/image_features.py) / [C++](cpp/basic/image_features.cpp) |
| Q54 | Gabor特征提取 | ⭐⭐⭐ | [Python](python/basic/image_features.py) / [C++](cpp/basic/image_features.cpp) |
| Q55 | 颜色直方图 | ⭐⭐ | [Python](python/basic/image_features.py) / [C++](cpp/basic/image_features.cpp) |

### 12. 图像匹配 (image_matching.py)
| 问题编号 | 问题名称 | 难度等级 | 代码索引 |
|----------|----------|----------|----------|
| Q56 | 模板匹配(SSD) | ⭐⭐ | [Python](python/basic/image_matching.py) / [C++](cpp/basic/image_matching.cpp) |
| Q57 | 模板匹配(SAD) | ⭐⭐ | [Python](python/basic/image_matching.py) / [C++](cpp/basic/image_matching.cpp) |
| Q58 | 模板匹配(NCC) | ⭐⭐ | [Python](python/basic/image_matching.py) / [C++](cpp/basic/image_matching.cpp) |
| Q59 | 模板匹配(ZNCC) | ⭐⭐⭐ | [Python](python/basic/image_matching.py) / [C++](cpp/basic/image_matching.cpp) |
| Q60 | 特征点匹配 | ⭐⭐⭐ | [Python](python/basic/image_matching.py) / [C++](cpp/basic/image_matching.cpp) |

### 13. 连通域分析 (connected_components.py)
| 问题编号 | 问题名称 | 难度等级 | 代码索引 |
|----------|----------|----------|----------|
| Q61 | 4连通域标记 | ⭐⭐ | [Python](python/basic/connected_components.py) / [C++](cpp/basic/connected_components.cpp) |
| Q62 | 8连通域标记 | ⭐⭐ | [Python](python/basic/connected_components.py) / [C++](cpp/basic/connected_components.cpp) |
| Q63 | 连通域统计 | ⭐⭐ | [Python](python/basic/connected_components.py) / [C++](cpp/basic/connected_components.cpp) |
| Q64 | 连通域过滤 | ⭐⭐ | [Python](python/basic/connected_components.py) / [C++](cpp/basic/connected_components.cpp) |
| Q65 | 连通域属性计算 | ⭐⭐⭐ | [Python](python/basic/connected_components.py) / [C++](cpp/basic/connected_components.cpp) |

### 14. 图像细化 (thinning.py)
| 问题编号 | 问题名称 | 难度等级 | 代码索引 |
|----------|----------|----------|----------|
| Q66 | 基本细化算法 | ⭐⭐⭐ | [Python](python/basic/thinning.py) / [C++](cpp/basic/thinning.cpp) |
| Q67 | Hilditch细化 | ⭐⭐⭐ | [Python](python/basic/thinning.py) / [C++](cpp/basic/thinning.cpp) |
| Q68 | Zhang-Suen细化 | ⭐⭐⭐ | [Python](python/basic/thinning.py) / [C++](cpp/basic/thinning.cpp) |
| Q69 | 骨架提取 | ⭐⭐⭐ | [Python](python/basic/thinning.py) / [C++](cpp/basic/thinning.cpp) |
| Q70 | 中轴变换 | ⭐⭐⭐ | [Python](python/basic/thinning.py) / [C++](cpp/basic/thinning.cpp) |

### 15. 目标检测 (object_detection.py)
| 问题编号 | 问题名称 | 难度等级 | 代码索引 |
|----------|----------|----------|----------|
| Q71 | 滑动窗口检测 | ⭐⭐⭐ | [Python](python/basic/object_detection.py) / [C++](cpp/basic/object_detection.cpp) |
| Q72 | HOG+SVM检测 | ⭐⭐⭐ | [Python](python/basic/object_detection.py) / [C++](cpp/basic/object_detection.cpp) |
| Q73 | Haar+AdaBoost检测 | ⭐⭐⭐ | [Python](python/basic/object_detection.py) / [C++](cpp/basic/object_detection.cpp) |
| Q74 | 非极大值抑制 | ⭐⭐⭐ | [Python](python/basic/object_detection.py) / [C++](cpp/basic/object_detection.cpp) |
| Q75 | 目标跟踪 | ⭐⭐⭐ | [Python](python/basic/object_detection.py) / [C++](cpp/basic/object_detection.cpp) |

### 16. 图像金字塔 (image_pyramid.py)
| 问题编号 | 问题名称 | 难度等级 | 代码索引 |
|----------|----------|----------|----------|
| Q76 | 高斯金字塔 | ⭐⭐ | [Python](python/basic/image_pyramid.py) / [C++](cpp/basic/image_pyramid.cpp) |
| Q77 | 拉普拉斯金字塔 | ⭐⭐⭐ | [Python](python/basic/image_pyramid.py) / [C++](cpp/basic/image_pyramid.cpp) |
| Q78 | 图像融合 | ⭐⭐⭐ | [Python](python/basic/image_pyramid.py) / [C++](cpp/basic/image_pyramid.cpp) |
| Q79 | SIFT尺度空间 | ⭐⭐⭐ | [Python](python/basic/image_pyramid.py) / [C++](cpp/basic/image_pyramid.cpp) |
| Q80 | 显著性检测 | ⭐⭐⭐ | [Python](python/basic/image_pyramid.py) / [C++](cpp/basic/image_pyramid.cpp) |

### 17. 纹理分析 (texture_analysis.py)
| 问题编号 | 问题名称 | 难度等级 | 代码索引 |
|----------|----------|----------|----------|
| Q81 | 灰度共生矩阵 | ⭐⭐⭐ | [Python](python/basic/texture_analysis.py) / [C++](cpp/basic/texture_analysis.cpp) |
| Q82 | 纹理统计特征 | ⭐⭐⭐ | [Python](python/basic/texture_analysis.py) / [C++](cpp/basic/texture_analysis.cpp) |
| Q83 | 局部二值模式 | ⭐⭐⭐ | [Python](python/basic/texture_analysis.py) / [C++](cpp/basic/texture_analysis.cpp) |
| Q84 | Gabor纹理特征 | ⭐⭐⭐ | [Python](python/basic/texture_analysis.py) / [C++](cpp/basic/texture_analysis.cpp) |
| Q85 | 纹理分类 | ⭐⭐⭐ | [Python](python/basic/texture_analysis.py) / [C++](cpp/basic/texture_analysis.cpp) |

### 18. 图像修复 (image_inpainting.py)
| 问题编号 | 问题名称 | 难度等级 | 代码索引 |
|----------|----------|----------|----------|
| Q86 | 基于扩散的修复 | ⭐⭐⭐ | [Python](python/basic/image_inpainting.py) / [C++](cpp/basic/image_inpainting.cpp) |
| Q87 | 基于块匹配的修复 | ⭐⭐⭐ | [Python](python/basic/image_inpainting.py) / [C++](cpp/basic/image_inpainting.cpp) |
| Q88 | 基于PatchMatch的修复 | ⭐⭐⭐ | [Python](python/basic/image_inpainting.py) / [C++](cpp/basic/image_inpainting.cpp) |
| Q89 | 基于深度学习的修复 | ⭐⭐⭐ | [Python](python/basic/image_inpainting.py) / [C++](cpp/basic/image_inpainting.cpp) |
| Q90 | 视频修复 | ⭐⭐⭐ | [Python](python/basic/image_inpainting.py) / [C++](cpp/basic/image_inpainting.cpp) |

### 19. 图像质量评价 (image_quality.py)
| 问题编号 | 问题名称 | 难度等级 | 代码索引 |
|----------|----------|----------|----------|
| Q91 | 峰值信噪比(PSNR) | ⭐⭐ | [Python](python/basic/image_quality.py) / [C++](cpp/basic/image_quality.cpp) |
| Q92 | 结构相似性(SSIM) | ⭐⭐⭐ | [Python](python/basic/image_quality.py) / [C++](cpp/basic/image_quality.cpp) |
| Q93 | 均方误差(MSE) | ⭐⭐ | [Python](python/basic/image_quality.py) / [C++](cpp/basic/image_quality.cpp) |
| Q94 | 视觉信息保真度(VIF) | ⭐⭐⭐ | [Python](python/basic/image_quality.py) / [C++](cpp/basic/image_quality.cpp) |
| Q95 | 无参考质量评价 | ⭐⭐⭐ | [Python](python/basic/image_quality.py) / [C++](cpp/basic/image_quality.cpp) |

### 20. 图像超分辨率 (super_resolution.py)
| 问题编号 | 问题名称 | 难度等级 | 代码索引 |
|----------|----------|----------|----------|
| Q96 | 双三次插值 | ⭐⭐ | [Python](python/basic/super_resolution.py) / [C++](cpp/basic/super_resolution.cpp) |
| Q97 | 基于稀疏表示的超分辨率 | ⭐⭐⭐ | [Python](python/basic/super_resolution.py) / [C++](cpp/basic/super_resolution.cpp) |
| Q98 | 基于深度学习的超分辨率 | ⭐⭐⭐ | [Python](python/basic/super_resolution.py) / [C++](cpp/basic/super_resolution.cpp) |
| Q99 | 多帧超分辨率 | ⭐⭐⭐ | [Python](python/basic/super_resolution.py) / [C++](cpp/basic/super_resolution.cpp) |
| Q100 | 实时超分辨率 | ⭐⭐⭐ | [Python](python/basic/super_resolution.py) / [C++](cpp/basic/super_resolution.cpp) |

## 进阶算法列表

### 1. 图像增强算法
| 算法名称 | 类别 | Python | C++ | 难度等级 | 代码索引 |
|----------|------|---------|-----|----------|----------|
| Retinex MSRCR | 图像增强 | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/retinex_msrcr.py) / [C++](cpp/advanced/image_enhancement/retinex_msrcr.cpp) |
| HDR | 高动态范围 | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/hdr.py) / [C++](cpp/advanced/HDR.cpp) |
| 自适应对数映射 | 高动态范围 | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/adaptive_logarithmic_mapping.py) / [C++](cpp/advanced/image_enhancement/adaptive_logarithmic_mapping.cpp) |
| 多尺度细节增强 | 图像增强 | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/multi_scale_detail_enhancement.py) / [C++](cpp/advanced/image_enhancement/multi_scale_detail_enhancement.cpp) |
| 实时自适应对比度 | 图像增强 | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/real_time_adaptive_contrast.py) / [C++](cpp/advanced/image_enhancement/real_time_adaptive_contrast.cpp) |
| 自动色彩均衡(ACE) | 色彩增强 | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/automatic_color_equalization.py) / [C++](cpp/advanced/image_enhancement/automatic_color_equalization.cpp) |

### 2. 图像矫正算法
| 算法名称 | 类别 | Python | C++ | 难度等级 | 代码索引 |
|----------|------|---------|-----|----------|----------|
| 自动白平衡 | 色彩校正 | ✅ | ✅ | ⭐⭐ | [Python](python/advanced/automatic_white_balance.py) / [C++](cpp/advanced/image_correction/automatic_white_balance.cpp) |
| 自动色阶调整 | 色彩校正 | ✅ | ✅ | ⭐⭐ | [Python](python/advanced/auto_level_adjustment.py) / [C++](cpp/advanced/image_correction/auto_level.cpp) |
| 光照不均匀校正 | 图像矫正 | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/illumination_correction.py) / [C++](cpp/advanced/image_correction/illumination_correction.cpp) |
| 逆光图像恢复 | 图像恢复 | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/backlight_correction.py) / [C++](cpp/advanced/image_correction/backlight.cpp) |
| 二维伽马校正 | 图像矫正 | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/gamma_correction.py) / [C++](cpp/advanced/image_correction/gamma_correction.cpp) |

### 3. 图像去雾算法
| 算法名称 | 类别 | Python | C++ | 难度等级 | 代码索引 |
|----------|------|---------|-----|----------|----------|
| 暗通道去雾 | 图像去雾 | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/dark_channel_dehazing.py) / [C++](cpp/advanced/image_defogging/dark_channel.cpp) |
| 导向滤波去雾 | 图像去雾 | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/guided_filter.py) / [C++](cpp/advanced/image_defogging/guided_filter.cpp) |
| 中值滤波去雾 | 图像去雾 | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/median_filter_defogging.py) / [C++](cpp/advanced/image_defogging/median_filter.cpp) |
| 快速单图去雾 | 图像去雾 | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/fast_defogging.py) / [C++](cpp/advanced/image_defogging/fast_defogging.cpp) |
| 实时视频去雾 | 图像去雾 | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/realtime_dehazing.py) / [C++](cpp/advanced/image_defogging/realtime_dehazing.cpp) |

### 4. 高级滤波算法
| 算法名称 | 类别 | Python | C++ | 难度等级 | 代码索引 |
|----------|------|---------|-----|----------|----------|
| 导向滤波 | 图像滤波 | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/guided_filter.py) / [C++](cpp/advanced/advanced_filtering/guided_filter.cpp) |
| 侧窗口滤波(Box) | 图像滤波 | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/side_window_filter.py) / [C++](cpp/advanced/advanced_filtering/side_window_filter.cpp) |
| 侧窗口滤波(Median) | 图像滤波 | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/side_window_filter.py) / [C++](cpp/advanced/advanced_filtering/side_window_filter.cpp) |
| 同态滤波 | 图像滤波 | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/homomorphic_filter.py) / [C++](cpp/advanced/advanced_filtering/homomorphic_filter.cpp) |

### 5. 特殊目标检测
| 算法名称 | 类别 | Python | C++ | 难度等级 | 代码索引 |
|----------|------|---------|-----|----------|----------|
| 矩形检测 | 目标检测 | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/rectangle_detection.py) / [C++](cpp/advanced/special_detection/rectangle_detection.cpp) |
| 车牌检测 | 目标检测 | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/license_plate_detection.py) / [C++](cpp/advanced/special_detection/license_plate_detection.cpp) |
| 偏色检测 | 图像检测 | ✅ | ✅ | ⭐⭐ | [Python](python/advanced/color_cast_detection.py) / [C++](cpp/advanced/special_detection/color_cast_detection.cpp) |

### 6. 图像特效算法
| 算法名称 | 类别 | Python | C++ | 难度等级 | 代码索引 |
|----------|------|---------|-----|----------|----------|
| 老照片特效 | 图像特效 | ✅ | ✅ | ⭐⭐ | [Python](python/advanced/vintage_effect.py) / [C++](cpp/advanced/image_effects/vintage_effect.cpp) |
| 运动模糊 | 图像特效 | ✅ | ✅ | ⭐⭐ | [Python](python/advanced/motion_blur_effect.py) / [C++](cpp/advanced/image_effects/motion_blur.cpp) |
| 球面化效果 | 图像特效 | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/spherize_effect.py) / [C++](cpp/advanced/image_effects/spherize.cpp) |
| 磨皮美白 | 图像特效 | ✅ | ✅ | ⭐⭐ | [Python](python/advanced/skin_beauty.py) / [C++](cpp/advanced/image_effects/skin_beauty.cpp) |
| 钝化蒙版 | 图像特效 | ✅ | ✅ | ⭐⭐ | [Python](python/advanced/unsharp_masking.py) / [C++](cpp/advanced/image_effects/unsharp_masking.cpp) |
| 油画效果 | 图像特效 | ✅ | ✅ | ⭐⭐ | [Python](python/advanced/oil_painting_effect.py) / [C++](cpp/advanced/image_effects/oil_painting_effect.cpp) |
| 卡通效果 | 图像特效 | ✅ | ✅ | ⭐⭐ | [Python](python/advanced/cartoon_effect.py) / [C++](cpp/advanced/image_effects/cartoon_effect.cpp) |

## 📚 文档与教程

### 📖 算法文档 (`docs/algorithms/`)
详细的技术文档，包含：
- **算法原理**：数学推导和核心思想
- **复杂度分析**：时间和空间复杂度评估
- **应用场景**：实际使用案例和最佳实践
- **参数调优**：算法参数选择和优化建议

### 🎓 学习教程 (`docs/tutorials/`)
从入门到精通的完整学习路径：
- **环境配置**：开发环境搭建指南
- **基础入门**：图像处理基础概念
- **实践案例**：手把手算法实现教程
- **性能分析**：代码性能评估和优化

### ⚡ 性能优化 (`docs/optimization/`)
高级优化技术指南：
- **多线程并行**：OpenMP和线程池优化
- **SIMD向量化**：CPU指令集优化
- **内存管理**：缓存友好的数据结构
- **算法改进**：数学优化和近似算法

## 使用说明

1. 克隆项目到本地：
```bash
git clone https://github.com/GlimmerLab/IP101.git
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 选择编程语言：
   - Python用户：直接运行Python文件
   - C++用户：需要先配置OpenCV环境

4. 运行示例：
```bash
# 基础问题示例
python python/basic/color_operations.py 1  # 运行颜色操作中的问题1
python python/basic/filtering.py 6         # 运行滤波中的问题6

# 进阶算法示例
python python/advanced/retinex_msrcr.py
```

### C++项目构建与使用

1. 环境配置:
   - 安装C++编译器 (如GCC, Visual Studio, Clang)
   - 安装CMake (3.10或更高版本)
   - 安装OpenCV库 (推荐4.0或更高版本)

2. 构建项目:
```bash
# 在项目根目录创建build文件夹
mkdir build && cd build

# 使用CMake配置（自动查找OpenCV）
cmake ..

# 或手动指定OpenCV路径
cmake -DOPENCV_DIR=/path/to/opencv/build ..

# 编译
cmake --build . --config Release
```

   **提示**：您也可以直接在主CMakeLists.txt文件中设置OpenCV路径：
   ```cmake
   # 打开CMakeLists.txt，找到以下部分并取消注释，修改为您的OpenCV路径
   # set(OpenCV_DIR "D:/opencv/build")    # Windows示例路径
   # set(OpenCV_DIR "/usr/local/opencv4") # Linux示例路径
   ```

3. 运行C++示例:
```bash
# 运行基础示例
./examples/basic_example

# 运行特定算法测试
./examples/basic/color_operations_test
./examples/basic/filtering_test
```

**提示**：
- 确保在正确的构建目录下执行可执行文件（如 `build/Release/` 或 `build/Debug/`）
- **Windows用户特别注意**：需要将OpenCV的DLL文件（如 `opencv_world4xx.dll`）复制到可执行文件所在目录，或添加到系统PATH环境变量中
- 如果遇到"找不到opencv_world4xx.dll"错误，请检查OpenCV的bin目录是否在PATH中，或手动复制DLL文件到程序目录

4. 开发自己的应用:
```cpp
// my_app.cpp
#include <opencv2/opencv.hpp>
#include <basic/color_operations.hpp>
#include <basic/filtering.hpp>

int main() {
    cv::Mat image = cv::imread("your_image.jpg");
    cv::Mat gray, filtered;

    // 使用库中的灰度转换函数
    ip101::to_gray(image, gray);

    // 使用高斯滤波
    ip101::gaussian_filter(gray, filtered, 3, 1.0);

    cv::imshow("Filtered Image", filtered);
    cv::waitKey(0);

    return 0;
}
```

5. 编译自定义应用:
```bash
g++ -std=c++17 my_app.cpp -o my_app -I/path/to/IP101/include `pkg-config --cflags --libs opencv4`
```

## ❓ 常见问题

### Q: 为什么选择手动实现而不是使用OpenCV内置函数？
A: 手动实现有助于深入理解算法原理，这是本项目的教学特色。在实际项目中，您可以根据需要选择使用OpenCV内置函数。

### Q: 如何选择合适的编程语言？
A:
- **Python**: 适合快速原型开发和算法验证
- **C++**: 适合性能要求高的生产环境

### Q: Windows下遇到DLL错误怎么办？
A: 请参考[使用说明](#使用说明)中的Windows特定提示，确保OpenCV DLL文件在正确位置。

### Q: 如何贡献代码？
A: 欢迎提交Issue和Pull Request！请参考下面的贡献指南。

## 🤝 贡献指南

1. Fork 本仓库
2. 创建新的分支: `git checkout -b feature/your-feature`
3. 提交更改: `git commit -am 'Add some feature'`
4. 推送到分支: `git push origin feature/your-feature`
5. 提交 Pull Request

### 贡献类型
- 🐛 Bug修复
- ✨ 新功能添加
- 📚 文档改进
- 🎨 代码优化
- 🧪 测试用例

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 致谢

IP101 参考和借鉴了下列项目：

### 🖼️ 图像处理算法参考
- [BBuf/Image-processing-algorithm](https://github.com/BBuf/Image-processing-algorithm)
- [gzr2017/ImageProcessing100Wen](https://github.com/gzr2017/ImageProcessing100Wen)
- [KuKuXia/Image_Processing_100_Questions](https://github.com/KuKuXia/Image_Processing_100_Questions)
- [ryoppippi/Gasyori100knock](https://github.com/ryoppippi/Gasyori100knock)

### 🔧 核心依赖库
- [OpenCV](https://github.com/opencv/opencv) - 计算机视觉库
- [scikit-image](https://github.com/scikit-image/scikit-image) - Python图像处理库
- [SimpleCV](https://github.com/sightmachine/SimpleCV) - 计算机视觉框架

### 🖥️ GUI界面依赖
- [GLFW](https://github.com/glfw/glfw) - 跨平台OpenGL上下文和窗口管理库
- [Dear ImGui](https://github.com/ocornut/imgui) - 轻量级即时模式图形用户界面库

### 📚 学习资源
- [imageshop](https://www.cnblogs.com/imageshop) - 图像处理技术博客
