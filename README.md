# IP101 - 图像处理基础100问

[English](README_EN.md) | 简体中文

IP101 是一个专注于图像处理基础知识、操作、应用和优化的系列教程。本系列旨在通过100个精心设计的问题，帮助读者全面掌握图像处理的核心概念和实践技能。

## 项目结构

```
IP101/
├── docs/                # 文档目录
│   ├── basic/          # 基础问题文档
│   │   ├── color_operations.md      # 颜色操作
│   │   ├── filtering.md             # 图像滤波
│   │   ├── edge_detection.md        # 边缘检测
│   │   ├── image_transform.md       # 图像变换
│   │   ├── image_enhancement.md     # 图像增强
│   │   ├── feature_extraction.md    # 特征提取
│   │   ├── image_segmentation.md    # 图像分割
│   │   ├── morphology.md            # 形态学处理
│   │   ├── frequency_domain.md      # 频域处理
│   │   ├── image_compression.md     # 图像压缩
│   │   ├── image_features.md        # 图像特征
│   │   ├── image_matching.md        # 图像匹配
│   │   ├── connected_components.md  # 连通域分析
│   │   ├── thinning.md             # 图像细化
│   │   ├── object_detection.md      # 目标检测
│   │   ├── image_pyramid.md         # 图像金字塔
│   │   ├── texture_analysis.md      # 纹理分析
│   │   ├── image_inpainting.md      # 图像修复
│   │   ├── image_quality.md         # 图像质量评价
│   │   └── super_resolution.md      # 超分辨率
│   └── advanced/       # 进阶算法文档
│       ├── image_enhancement/        # 图像增强算法
│       ├── image_correction/         # 图像矫正算法
│       ├── image_dehazing/          # 图像去雾算法
│       ├── advanced_filtering/       # 高级滤波算法
│       ├── special_detection/        # 特殊目标检测
│       └── image_effects/           # 图像特效算法
├── python/             # Python实现
│   ├── basic/          # 基础问题代码
│   │   ├── color_operations.py      # 颜色操作相关算法
│   │   ├── filtering.py             # 图像滤波相关算法
│   │   ├── edge_detection.py        # 边缘检测相关算法
│   │   ├── image_transform.py       # 图像变换相关算法
│   │   ├── image_enhancement.py     # 图像增强相关算法
│   │   ├── feature_extraction.py    # 特征提取相关算法
│   │   ├── image_segmentation.py    # 图像分割相关算法
│   │   ├── morphology.py            # 形态学处理相关算法
│   │   ├── frequency_domain.py      # 频域处理相关算法
│   │   ├── image_compression.py     # 图像压缩相关算法
│   │   ├── image_features.py        # 图像特征相关算法
│   │   ├── image_matching.py        # 图像匹配相关算法
│   │   ├── connected_components.py  # 连通域分析相关算法
│   │   ├── thinning.py             # 图像细化相关算法
│   │   ├── object_detection.py      # 目标检测相关算法
│   │   ├── image_pyramid.py         # 图像金字塔相关算法
│   │   ├── texture_analysis.py      # 纹理分析相关算法
│   │   ├── image_inpainting.py      # 图像修复相关算法
│   │   ├── image_quality.py         # 图像质量评价相关算法
│   │   └── super_resolution.py      # 超分辨率相关算法
│   └── advanced/       # 进阶算法代码
│       ├── image_enhancement/        # 图像增强算法
│       ├── image_correction/         # 图像矫正算法
│       ├── image_dehazing/          # 图像去雾算法
│       ├── advanced_filtering/       # 高级滤波算法
│       ├── special_detection/        # 特殊目标检测
│       └── image_effects/           # 图像特效算法
├── cpp/                # C++实现
│   ├── basic/          # 基础问题代码
│   │   ├── color_operations.cpp     # 颜色操作相关算法
│   │   ├── filtering.cpp            # 图像滤波相关算法
│   │   ├── edge_detection.cpp       # 边缘检测相关算法
│   │   ├── image_transform.cpp      # 图像变换相关算法
│   │   ├── image_enhancement.cpp    # 图像增强相关算法
│   │   ├── feature_extraction.cpp   # 特征提取相关算法
│   │   ├── image_segmentation.cpp   # 图像分割相关算法
│   │   ├── morphology.cpp           # 形态学处理相关算法
│   │   ├── frequency_domain.cpp     # 频域处理相关算法
│   │   ├── image_compression.cpp    # 图像压缩相关算法
│   │   ├── image_features.cpp       # 图像特征相关算法
│   │   ├── image_matching.cpp       # 图像匹配相关算法
│   │   ├── connected_components.cpp # 连通域分析相关算法
│   │   ├── thinning.cpp            # 图像细化相关算法
│   │   ├── object_detection.cpp     # 目标检测相关算法
│   │   ├── image_pyramid.cpp        # 图像金字塔相关算法
│   │   ├── texture_analysis.cpp     # 纹理分析相关算法
│   │   ├── image_inpainting.cpp     # 图像修复相关算法
│   │   ├── image_quality.cpp        # 图像质量评价相关算法
│   │   └── super_resolution.cpp     # 超分辨率相关算法
│   └── advanced/       # 进阶算法代码
│       ├── image_enhancement/        # 图像增强算法
│       ├── image_correction/         # 图像矫正算法
│       ├── image_dehazing/          # 图像去雾算法
│       ├── advanced_filtering/       # 高级滤波算法
│       ├── special_detection/        # 特殊目标检测
│       └── image_effects/           # 图像特效算法
├── examples/           # 示例代码
│   ├── basic/          # 基础问题示例
│   └── advanced/       # 进阶算法示例
└── images/             # 示例图片
```

## 内容特点

- 基础到进阶的完整学习路径
- 理论与实践相结合
- 丰富的代码示例
- 实际应用场景分析
- 性能优化技巧
- **实现方式**：所有算法均为手动实现，不使用OpenCV内置函数，帮助理解算法原理

## 支持的语言

| 语言 | 状态 | 说明 |
|------|------|------|
| Python | ✅ | 完整支持，包含所有100个问题的解答 |
| C++ | ✅ | 完整支持，包含所有100个问题的解答 |
| MATLAB | ❌ | 暂不支持 |
| Java | ❌ | 暂不支持 |

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
| Retinex MSRCR | 图像增强 | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/retinex_msrcr.py) / [C++](cpp/advanced/Retinex_MSRCR.cpp) |
| HDR | 高动态范围 | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/hdr.py) / [C++](cpp/advanced/HDR.cpp) |
| 自适应对数映射 | 高动态范围 | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/adaptive_logarithmic.py) / [C++](cpp/advanced/AdaptiveLogarithmicMapping.cpp) |
| 多尺度细节增强 | 图像增强 | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/detail_boosting.py) / [C++](cpp/advanced/MultiScaleDetailBoosting.cpp) |
| 实时自适应对比度 | 图像增强 | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/adaptive_contrast.py) / [C++](cpp/advanced/RealTimeAdaptiveContrast.cpp) |
| 自动色彩均衡(ACE) | 色彩增强 | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/ace.py) / [C++](cpp/advanced/AutomaticColorEqualization.cpp) |

### 2. 图像矫正算法
| 算法名称 | 类别 | Python | C++ | 难度等级 | 代码索引 |
|----------|------|---------|-----|----------|----------|
| 自动白平衡 | 色彩校正 | ✅ | ✅ | ⭐⭐ | [Python](python/advanced/auto_white_balance.py) / [C++](cpp/advanced/AutomaticWhiteBalanceMethod.cpp) |
| 自动色阶调整 | 色彩校正 | ✅ | ✅ | ⭐⭐ | [Python](python/advanced/auto_level.py) / [C++](cpp/advanced/AutoLevelAndAutoContrast.cpp) |
| 光照不均匀校正 | 图像矫正 | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/illumination_correction.py) / [C++](cpp/advanced/IlluminationCorrection.cpp) |
| 逆光图像恢复 | 图像恢复 | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/backlight.py) / [C++](cpp/advanced/Inrbl.cpp) |
| 二维伽马校正 | 图像矫正 | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/gamma_correction.py) / [C++](cpp/advanced/TwoDimensionalGamma.cpp) |

### 3. 图像去雾算法
| 算法名称 | 类别 | Python | C++ | 难度等级 | 代码索引 |
|----------|------|---------|-----|----------|----------|
| 暗通道去雾 | 图像去雾 | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/dark_channel.py) / [C++](cpp/advanced/DarkChannelPrior.cpp) |
| 导向滤波去雾 | 图像去雾 | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/guided_filter.py) / [C++](cpp/advanced/GuidedFilterDehazing.cpp) |
| 中值滤波去雾 | 图像去雾 | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/median_filter.py) / [C++](cpp/advanced/MedianFilterFogRemoval.cpp) |
| 快速单图去雾 | 图像去雾 | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/fast_defogging.py) / [C++](cpp/advanced/FastDefogging.cpp) |
| 实时视频去雾 | 图像去雾 | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/realtime_dehazing.py) / [C++](cpp/advanced/RealtimeDehazing.cpp) |

### 4. 高级滤波算法
| 算法名称 | 类别 | Python | C++ | 难度等级 | 代码索引 |
|----------|------|---------|-----|----------|----------|
| 导向滤波 | 图像滤波 | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/guided_filter.py) / [C++](cpp/advanced/GuidedFilter.cpp) |
| 侧窗口滤波(Box) | 图像滤波 | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/side_window.py) / [C++](cpp/advanced/BoxSideWindowFilter.cpp) |
| 侧窗口滤波(Median) | 图像滤波 | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/side_window.py) / [C++](cpp/advanced/MedianSideWindowFilter.cpp) |
| 同态滤波 | 图像滤波 | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/homomorphic.py) / [C++](cpp/advanced/HomomorphicFilter.cpp) |

### 5. 特殊目标检测
| 算法名称 | 类别 | Python | C++ | 难度等级 | 代码索引 |
|----------|------|---------|-----|----------|----------|
| 矩形检测 | 目标检测 | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/rectangle_detection.py) / [C++](cpp/advanced/RectangleDetection.cpp) |
| 车牌检测 | 目标检测 | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/license_plate.py) / [C++](cpp/advanced/LicensePlateDetection.cpp) |
| 偏色检测 | 图像检测 | ✅ | ✅ | ⭐⭐ | [Python](python/advanced/color_cast.py) / [C++](cpp/advanced/ColorCastDetection.cpp) |

### 6. 图像特效算法
| 算法名称 | 类别 | Python | C++ | 难度等级 | 代码索引 |
|----------|------|---------|-----|----------|----------|
| 老照片特效 | 图像特效 | ✅ | ✅ | ⭐⭐ | [Python](python/advanced/vintage_effect.py) / [C++](cpp/advanced/VintageEffect.cpp) |
| 运动模糊 | 图像特效 | ✅ | ✅ | ⭐⭐ | [Python](python/advanced/motion_blur.py) / [C++](cpp/advanced/MotionBlur.cpp) |
| 球面化效果 | 图像特效 | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/spherize.py) / [C++](cpp/advanced/Spherize.cpp) |
| 磨皮美白 | 图像特效 | ✅ | ✅ | ⭐⭐ | [Python](python/advanced/skin_beauty.py) / [C++](cpp/advanced/SkinBeauty.cpp) |
| 钝化蒙版 | 图像特效 | ✅ | ✅ | ⭐⭐ | [Python](python/advanced/usm.py) / [C++](cpp/advanced/UnsharpMasking.cpp) |

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

## 进阶内容说明

1. 图像矫正算法
   - 自动色阶调整
   - 自动对比度调整
   - 光照不均匀校正

2. 图像滤波算法
   - 中值滤波
   - 导向滤波
   - 双边滤波

3. 特征提取算法
   - SIFT特征
   - SURF特征
   - ORB特征

4. 色彩空间转换
   - RGB转HSV
   - RGB转LAB
   - RGB转YUV

5. 算法优化
   - 多线程优化
   - GPU加速
   - SIMD优化

## 贡献指南

1. Fork 本仓库
2. 创建新的分支: `git checkout -b feature/your-feature`
3. 提交更改: `git commit -am 'Add some feature'`
4. 推送到分支: `git push origin feature/your-feature`
5. 提交 Pull Request

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 致谢

IP101 参考和借鉴了下列项目：

- [BBuf/Image-processing-algorithm](https://github.com/BBuf/Image-processing-algorithm)

- [gzr2017/ImageProcessing100Wen](https://github.com/gzr2017/ImageProcessing100Wen)

- [KuKuXia/Image_Processing_100_Questions](https://github.com/KuKuXia/Image_Processing_100_Questions)

- [ryoppippi/Gasyori100knock](https://github.com/ryoppippi/Gasyori100knock)

- [OpenCV](https://github.com/opencv/opencv)

- [scikit-image](https://github.com/scikit-image/scikit-image)

- [SimpleCV](https://github.com/sightmachine/SimpleCV)
