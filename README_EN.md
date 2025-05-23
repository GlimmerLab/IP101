# IP101 - 100 Questions in Image Processing

English | [简体中文](README.md)

IP101 is a comprehensive tutorial series focused on fundamental knowledge, operations, applications, and optimization in image processing. This series aims to help readers master core concepts and practical skills in image processing through 100 carefully designed questions.

## Project Structure

```
IP101/
├── include/            # Header files
│   └── basic/         # Basic algorithm headers
├── docs/               # Documentation
│   ├── algorithms/    # Algorithm documentation
│   ├── tutorials/     # Tutorials
│   └── optimization/  # Optimization techniques
├── python/             # Python implementation
│   ├── basic/          # Basic questions code
│   │   ├── color_operations.py      # Color operations algorithms
│   │   ├── filtering.py             # Image filtering algorithms
│   │   ├── edge_detection.py        # Edge detection algorithms
│   │   └── ...                      # Other algorithms
│   └── advanced/       # Advanced algorithms code
├── cpp/                # C++ implementation
│   ├── basic/          # Basic questions code
│   │   ├── color_operations.cpp     # Color operations algorithms
│   │   ├── filtering.cpp            # Image filtering algorithms
│   │   ├── edge_detection.cpp       # Edge detection algorithms
│   │   └── ...                      # Other algorithms
│   └── advanced/       # Advanced algorithms code
├── examples/           # Example code
│   ├── basic/          # Basic questions examples
│   └── advanced/       # Advanced algorithms examples
├── assets/             # Asset files
└── tests/              # Test code
```

## Features

- Complete learning path from basics to advanced
- Integration of theory and practice
- Rich code examples
- Real application scenario analysis
- Performance optimization techniques
- **Teaching-oriented implementation**: All algorithms are manually implemented without using OpenCV built-in functions to help understand algorithm principles

## Supported Languages

| Language | Status | Description |
|----------|---------|------------|
| Python | ✅ | Full support, includes solutions for all 100 questions |
| C++ | ✅ | Full support, includes solutions for all 100 questions |
| MATLAB | ❌ | Not supported yet |
| Java | ❌ | Not supported yet |

## Basic Questions Categories

### 1. Color Operations (color_operations.py)
| Question No. | Name | Difficulty | Code Reference |
|-------------|------|------------|----------------|
| Q1 | Channel Swap | ⭐ | [Python](python/basic/color_operations.py) / [C++](cpp/basic/color_operations.cpp) |
| Q2 | Grayscale | ⭐ | [Python](python/basic/color_operations.py) / [C++](cpp/basic/color_operations.cpp) |
| Q3 | Thresholding | ⭐ | [Python](python/basic/color_operations.py) / [C++](cpp/basic/color_operations.cpp) |
| Q4 | Otsu's Method | ⭐⭐ | [Python](python/basic/color_operations.py) / [C++](cpp/basic/color_operations.cpp) |
| Q5 | HSV Transform | ⭐⭐ | [Python](python/basic/color_operations.py) / [C++](cpp/basic/color_operations.cpp) |

### 2. Image Filtering (filtering.py)
| Question No. | Name | Difficulty | Code Reference |
|-------------|------|------------|----------------|
| Q6 | Mean Filter | ⭐ | [Python](python/basic/filtering.py) / [C++](cpp/basic/filtering.cpp) |
| Q7 | Median Filter | ⭐⭐ | [Python](python/basic/filtering.py) / [C++](cpp/basic/filtering.cpp) |
| Q8 | Gaussian Filter | ⭐⭐ | [Python](python/basic/filtering.py) / [C++](cpp/basic/filtering.cpp) |
| Q9 | Mean Pooling | ⭐ | [Python](python/basic/filtering.py) / [C++](cpp/basic/filtering.cpp) |
| Q10 | Max Pooling | ⭐ | [Python](python/basic/filtering.py) / [C++](cpp/basic/filtering.cpp) |

### 3. Edge Detection (edge_detection.py)
| Question No. | Name | Difficulty | Code Reference |
|-------------|------|------------|----------------|
| Q11 | Differential Filter | ⭐⭐ | [Python](python/basic/edge_detection.py) / [C++](cpp/basic/edge_detection.cpp) |
| Q12 | Sobel Filter | ⭐⭐ | [Python](python/basic/edge_detection.py) / [C++](cpp/basic/edge_detection.cpp) |
| Q13 | Prewitt Filter | ⭐⭐ | [Python](python/basic/edge_detection.py) / [C++](cpp/basic/edge_detection.cpp) |
| Q14 | Laplacian Filter | ⭐⭐ | [Python](python/basic/edge_detection.py) / [C++](cpp/basic/edge_detection.cpp) |
| Q15 | Emboss Effect | ⭐⭐ | [Python](python/basic/edge_detection.py) / [C++](cpp/basic/edge_detection.cpp) |
| Q16 | Edge Detection | ⭐⭐ | [Python](python/basic/edge_detection.py) / [C++](cpp/basic/edge_detection.cpp) |

### 4. Image Transformation (image_transform.py)
| Question No. | Name | Difficulty | Code Reference |
|-------------|------|------------|----------------|
| Q17 | Affine Transform | ⭐⭐ | [Python](python/basic/image_transform.py) / [C++](cpp/basic/image_transform.cpp) |
| Q18 | Perspective Transform | ⭐⭐ | [Python](python/basic/image_transform.py) / [C++](cpp/basic/image_transform.cpp) |
| Q19 | Rotation | ⭐⭐ | [Python](python/basic/image_transform.py) / [C++](cpp/basic/image_transform.cpp) |
| Q20 | Scaling | ⭐⭐ | [Python](python/basic/image_transform.py) / [C++](cpp/basic/image_transform.cpp) |
| Q21 | Translation | ⭐ | [Python](python/basic/image_transform.py) / [C++](cpp/basic/image_transform.cpp) |
| Q22 | Mirror | ⭐ | [Python](python/basic/image_transform.py) / [C++](cpp/basic/image_transform.cpp) |

### 5. Image Enhancement (image_enhancement.py)
| Question No. | Name | Difficulty | Code Reference |
|-------------|------|------------|----------------|
| Q23 | Histogram Equalization | ⭐⭐ | [Python](python/basic/image_enhancement.py) / [C++](cpp/basic/image_enhancement.cpp) |
| Q24 | Gamma Transform | ⭐⭐ | [Python](python/basic/image_enhancement.py) / [C++](cpp/basic/image_enhancement.cpp) |
| Q25 | Contrast Stretching | ⭐⭐ | [Python](python/basic/image_enhancement.py) / [C++](cpp/basic/image_enhancement.cpp) |
| Q26 | Brightness Adjustment | ⭐ | [Python](python/basic/image_enhancement.py) / [C++](cpp/basic/image_enhancement.cpp) |
| Q27 | Saturation Adjustment | ⭐⭐ | [Python](python/basic/image_enhancement.py) / [C++](cpp/basic/image_enhancement.cpp) |

### 6. Feature Extraction (feature_extraction.py)
| Question No. | Name | Difficulty | Code Reference |
|-------------|------|------------|----------------|
| Q28 | Harris Corner Detection | ⭐⭐⭐ | [Python](python/basic/feature_extraction.py) / [C++](cpp/basic/feature_extraction.cpp) |
| Q29 | SIFT Features | ⭐⭐⭐ | [Python](python/basic/feature_extraction.py) / [C++](cpp/basic/feature_extraction.cpp) |
| Q30 | SURF Features | ⭐⭐⭐ | [Python](python/basic/feature_extraction.py) / [C++](cpp/basic/feature_extraction.cpp) |
| Q31 | ORB Features | ⭐⭐⭐ | [Python](python/basic/feature_extraction.py) / [C++](cpp/basic/feature_extraction.cpp) |
| Q32 | Feature Matching | ⭐⭐⭐ | [Python](python/basic/feature_extraction.py) / [C++](cpp/basic/feature_extraction.cpp) |

### 7. Image Segmentation (image_segmentation.py)
| Question No. | Name | Difficulty | Code Reference |
|-------------|------|------------|----------------|
| Q33 | Threshold Segmentation | ⭐⭐ | [Python](python/basic/image_segmentation.py) / [C++](cpp/basic/image_segmentation.cpp) |
| Q34 | K-means Segmentation | ⭐⭐⭐ | [Python](python/basic/image_segmentation.py) / [C++](cpp/basic/image_segmentation.cpp) |
| Q35 | Region Growing | ⭐⭐⭐ | [Python](python/basic/image_segmentation.py) / [C++](cpp/basic/image_segmentation.cpp) |
| Q36 | Watershed Segmentation | ⭐⭐⭐ | [Python](python/basic/image_segmentation.py) / [C++](cpp/basic/image_segmentation.cpp) |
| Q37 | Graph Cut Segmentation | ⭐⭐⭐ | [Python](python/basic/image_segmentation.py) / [C++](cpp/basic/image_segmentation.cpp) |

### 8. Morphological Operations (morphology.py)
| Question No. | Name | Difficulty | Code Reference |
|-------------|------|------------|----------------|
| Q38 | Dilation | ⭐⭐ | [Python](python/basic/morphology.py) / [C++](cpp/basic/morphology.cpp) |
| Q39 | Erosion | ⭐⭐ | [Python](python/basic/morphology.py) / [C++](cpp/basic/morphology.cpp) |
| Q40 | Opening | ⭐⭐ | [Python](python/basic/morphology.py) / [C++](cpp/basic/morphology.cpp) |
| Q41 | Closing | ⭐⭐ | [Python](python/basic/morphology.py) / [C++](cpp/basic/morphology.cpp) |
| Q42 | Morphological Gradient | ⭐⭐⭐ | [Python](python/basic/morphology.py) / [C++](cpp/basic/morphology.cpp) |

### 9. Frequency Domain Processing (frequency_domain.py)
| Question No. | Name | Difficulty | Code Reference |
|-------------|------|------------|----------------|
| Q43 | Fourier Transform | ⭐⭐⭐ | [Python](python/basic/frequency_domain.py) / [C++](cpp/basic/frequency_domain.cpp) |
| Q44 | Frequency Filtering | ⭐⭐⭐ | [Python](python/basic/frequency_domain.py) / [C++](cpp/basic/frequency_domain.cpp) |
| Q45 | DCT Transform | ⭐⭐⭐ | [Python](python/basic/frequency_domain.py) / [C++](cpp/basic/frequency_domain.cpp) |
| Q46 | Wavelet Transform | ⭐⭐⭐ | [Python](python/basic/frequency_domain.py) / [C++](cpp/basic/frequency_domain.cpp) |

### 10. Image Compression (image_compression.py)
| Question No. | Name | Difficulty | Code Reference |
|-------------|------|------------|----------------|
| Q47 | Lossless Compression | ⭐⭐⭐ | [Python](python/basic/image_compression.py) / [C++](cpp/basic/image_compression.cpp) |
| Q48 | JPEG Compression | ⭐⭐⭐ | [Python](python/basic/image_compression.py) / [C++](cpp/basic/image_compression.cpp) |
| Q49 | Fractal Compression | ⭐⭐⭐ | [Python](python/basic/image_compression.py) / [C++](cpp/basic/image_compression.cpp) |
| Q50 | Wavelet Compression | ⭐⭐⭐ | [Python](python/basic/image_compression.py) / [C++](cpp/basic/image_compression.cpp) |

### 11. Image Features (image_features.py)
| Question No. | Name | Difficulty | Code Reference |
|-------------|------|------------|----------------|
| Q51 | HOG Feature Extraction | ⭐⭐⭐ | [Python](python/basic/image_features.py) / [C++](cpp/basic/image_features.cpp) |
| Q52 | LBP Feature Extraction | ⭐⭐⭐ | [Python](python/basic/image_features.py) / [C++](cpp/basic/image_features.cpp) |
| Q53 | Haar Feature Extraction | ⭐⭐⭐ | [Python](python/basic/image_features.py) / [C++](cpp/basic/image_features.cpp) |
| Q54 | Gabor Feature Extraction | ⭐⭐⭐ | [Python](python/basic/image_features.py) / [C++](cpp/basic/image_features.cpp) |
| Q55 | Color Histogram | ⭐⭐ | [Python](python/basic/image_features.py) / [C++](cpp/basic/image_features.cpp) |

### 12. Image Matching (image_matching.py)
| Question No. | Name | Difficulty | Code Reference |
|-------------|------|------------|----------------|
| Q56 | Template Matching (SSD) | ⭐⭐ | [Python](python/basic/image_matching.py) / [C++](cpp/basic/image_matching.cpp) |
| Q57 | Template Matching (SAD) | ⭐⭐ | [Python](python/basic/image_matching.py) / [C++](cpp/basic/image_matching.cpp) |
| Q58 | Template Matching (NCC) | ⭐⭐ | [Python](python/basic/image_matching.py) / [C++](cpp/basic/image_matching.cpp) |
| Q59 | Template Matching (ZNCC) | ⭐⭐⭐ | [Python](python/basic/image_matching.py) / [C++](cpp/basic/image_matching.cpp) |
| Q60 | Feature Point Matching | ⭐⭐⭐ | [Python](python/basic/image_matching.py) / [C++](cpp/basic/image_matching.cpp) |

### 13. Connected Components Analysis (connected_components.py)
| Question No. | Name | Difficulty | Code Reference |
|-------------|------|------------|----------------|
| Q61 | 4-Connected Components | ⭐⭐ | [Python](python/basic/connected_components.py) / [C++](cpp/basic/connected_components.cpp) |
| Q62 | 8-Connected Components | ⭐⭐ | [Python](python/basic/connected_components.py) / [C++](cpp/basic/connected_components.cpp) |
| Q63 | Connected Components Statistics | ⭐⭐ | [Python](python/basic/connected_components.py) / [C++](cpp/basic/connected_components.cpp) |
| Q64 | Connected Components Filtering | ⭐⭐ | [Python](python/basic/connected_components.py) / [C++](cpp/basic/connected_components.cpp) |
| Q65 | Connected Components Properties | ⭐⭐⭐ | [Python](python/basic/connected_components.py) / [C++](cpp/basic/connected_components.cpp) |

### 14. Image Thinning (thinning.py)
| Question No. | Name | Difficulty | Code Reference |
|-------------|------|------------|----------------|
| Q66 | Basic Thinning | ⭐⭐⭐ | [Python](python/basic/thinning.py) / [C++](cpp/basic/thinning.cpp) |
| Q67 | Hilditch Thinning | ⭐⭐⭐ | [Python](python/basic/thinning.py) / [C++](cpp/basic/thinning.cpp) |
| Q68 | Zhang-Suen Thinning | ⭐⭐⭐ | [Python](python/basic/thinning.py) / [C++](cpp/basic/thinning.cpp) |
| Q69 | Skeleton Extraction | ⭐⭐⭐ | [Python](python/basic/thinning.py) / [C++](cpp/basic/thinning.cpp) |
| Q70 | Medial Axis Transform | ⭐⭐⭐ | [Python](python/basic/thinning.py) / [C++](cpp/basic/thinning.cpp) |

### 15. Object Detection (object_detection.py)
| Question No. | Name | Difficulty | Code Reference |
|-------------|------|------------|----------------|
| Q71 | Sliding Window Detection | ⭐⭐⭐ | [Python](python/basic/object_detection.py) / [C++](cpp/basic/object_detection.cpp) |
| Q72 | HOG+SVM Detection | ⭐⭐⭐ | [Python](python/basic/object_detection.py) / [C++](cpp/basic/object_detection.cpp) |
| Q73 | Haar+AdaBoost Detection | ⭐⭐⭐ | [Python](python/basic/object_detection.py) / [C++](cpp/basic/object_detection.cpp) |
| Q74 | Non-Maximum Suppression | ⭐⭐⭐ | [Python](python/basic/object_detection.py) / [C++](cpp/basic/object_detection.cpp) |
| Q75 | Object Tracking | ⭐⭐⭐ | [Python](python/basic/object_detection.py) / [C++](cpp/basic/object_detection.cpp) |

### 16. Image Pyramid (image_pyramid.py)
| Question No. | Name | Difficulty | Code Reference |
|-------------|------|------------|----------------|
| Q76 | Gaussian Pyramid | ⭐⭐ | [Python](python/basic/image_pyramid.py) / [C++](cpp/basic/image_pyramid.cpp) |
| Q77 | Laplacian Pyramid | ⭐⭐⭐ | [Python](python/basic/image_pyramid.py) / [C++](cpp/basic/image_pyramid.cpp) |
| Q78 | Image Blending | ⭐⭐⭐ | [Python](python/basic/image_pyramid.py) / [C++](cpp/basic/image_pyramid.cpp) |
| Q79 | SIFT Scale Space | ⭐⭐⭐ | [Python](python/basic/image_pyramid.py) / [C++](cpp/basic/image_pyramid.cpp) |
| Q80 | Saliency Detection | ⭐⭐⭐ | [Python](python/basic/image_pyramid.py) / [C++](cpp/basic/image_pyramid.cpp) |

### 17. Texture Analysis (texture_analysis.py)
| Question No. | Name | Difficulty | Code Reference |
|-------------|------|------------|----------------|
| Q81 | Gray Level Co-occurrence Matrix | ⭐⭐⭐ | [Python](python/basic/texture_analysis.py) / [C++](cpp/basic/texture_analysis.cpp) |
| Q82 | Texture Statistical Features | ⭐⭐⭐ | [Python](python/basic/texture_analysis.py) / [C++](cpp/basic/texture_analysis.cpp) |
| Q83 | Local Binary Pattern | ⭐⭐⭐ | [Python](python/basic/texture_analysis.py) / [C++](cpp/basic/texture_analysis.cpp) |
| Q84 | Gabor Texture Features | ⭐⭐⭐ | [Python](python/basic/texture_analysis.py) / [C++](cpp/basic/texture_analysis.cpp) |
| Q85 | Texture Classification | ⭐⭐⭐ | [Python](python/basic/texture_analysis.py) / [C++](cpp/basic/texture_analysis.cpp) |

### 18. Image Inpainting (image_inpainting.py)
| Question No. | Name | Difficulty | Code Reference |
|-------------|------|------------|----------------|
| Q86 | Diffusion-based Inpainting | ⭐⭐⭐ | [Python](python/basic/image_inpainting.py) / [C++](cpp/basic/image_inpainting.cpp) |
| Q87 | Patch-based Inpainting | ⭐⭐⭐ | [Python](python/basic/image_inpainting.py) / [C++](cpp/basic/image_inpainting.cpp) |
| Q88 | PatchMatch-based Inpainting | ⭐⭐⭐ | [Python](python/basic/image_inpainting.py) / [C++](cpp/basic/image_inpainting.cpp) |
| Q89 | Deep Learning-based Inpainting | ⭐⭐⭐ | [Python](python/basic/image_inpainting.py) / [C++](cpp/basic/image_inpainting.cpp) |
| Q90 | Video Inpainting | ⭐⭐⭐ | [Python](python/basic/image_inpainting.py) / [C++](cpp/basic/image_inpainting.cpp) |

### 19. Image Quality Assessment (image_quality.py)
| Question No. | Name | Difficulty | Code Reference |
|-------------|------|------------|----------------|
| Q91 | Peak Signal-to-Noise Ratio (PSNR) | ⭐⭐ | [Python](python/basic/image_quality.py) / [C++](cpp/basic/image_quality.cpp) |
| Q92 | Structural Similarity (SSIM) | ⭐⭐⭐ | [Python](python/basic/image_quality.py) / [C++](cpp/basic/image_quality.cpp) |
| Q93 | Mean Square Error (MSE) | ⭐⭐ | [Python](python/basic/image_quality.py) / [C++](cpp/basic/image_quality.cpp) |
| Q94 | Visual Information Fidelity (VIF) | ⭐⭐⭐ | [Python](python/basic/image_quality.py) / [C++](cpp/basic/image_quality.cpp) |
| Q95 | No-Reference Quality Assessment | ⭐⭐⭐ | [Python](python/basic/image_quality.py) / [C++](cpp/basic/image_quality.cpp) |

### 20. Super Resolution (super_resolution.py)
| Question No. | Name | Difficulty | Code Reference |
|-------------|------|------------|----------------|
| Q96 | Bicubic Interpolation | ⭐⭐ | [Python](python/basic/super_resolution.py) / [C++](cpp/basic/super_resolution.cpp) |
| Q97 | Sparse Representation-based SR | ⭐⭐⭐ | [Python](python/basic/super_resolution.py) / [C++](cpp/basic/super_resolution.cpp) |
| Q98 | Deep Learning-based SR | ⭐⭐⭐ | [Python](python/basic/super_resolution.py) / [C++](cpp/basic/super_resolution.cpp) |
| Q99 | Multi-frame SR | ⭐⭐⭐ | [Python](python/basic/super_resolution.py) / [C++](cpp/basic/super_resolution.cpp) |
| Q100 | Real-time SR | ⭐⭐⭐ | [Python](python/basic/super_resolution.py) / [C++](cpp/basic/super_resolution.cpp) |

## Advanced Content Overview

1. **Algorithm Documentation** - `docs/algorithms/`
   - Algorithm Principles
   - Mathematical Model Analysis
   - Algorithm Complexity Assessment
   - Real-world Application Scenarios

2. **Tutorials** - `docs/tutorials/`
   - Image Processing Introduction
   - Development Environment Setup
   - Algorithm Implementation Guides
   - Performance Optimization Methods

3. **Optimization Techniques** - `docs/optimization/`
   - Multi-threading Optimization
   - SIMD Vectorization
   - GPU Acceleration
   - Memory Optimization
   - Algorithm Improvements

## Advanced Algorithm List

### 1. Image Enhancement Algorithms
| Algorithm | Category | Python | C++ | Difficulty | Code Reference |
|-----------|----------|---------|-----|------------|----------------|
| Retinex MSRCR | Image Enhancement | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/retinex_msrcr.py) / [C++](cpp/advanced/Retinex_MSRCR.cpp) |
| HDR | High Dynamic Range | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/hdr.py) / [C++](cpp/advanced/HDR.cpp) |
| Adaptive Logarithmic Mapping | High Dynamic Range | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/adaptive_logarithmic.py) / [C++](cpp/advanced/AdaptiveLogarithmicMapping.cpp) |
| Multi-scale Detail Boosting | Image Enhancement | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/detail_boosting.py) / [C++](cpp/advanced/MultiScaleDetailBoosting.cpp) |
| Real-time Adaptive Contrast | Image Enhancement | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/adaptive_contrast.py) / [C++](cpp/advanced/RealTimeAdaptiveContrast.cpp) |
| Automatic Color Equalization | Color Enhancement | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/ace.py) / [C++](cpp/advanced/AutomaticColorEqualization.cpp) |

### 2. Image Correction Algorithms
| Algorithm | Category | Python | C++ | Difficulty | Code Reference |
|-----------|----------|---------|-----|------------|----------------|
| Auto White Balance | Color Correction | ✅ | ✅ | ⭐⭐ | [Python](python/advanced/auto_white_balance.py) / [C++](cpp/advanced/AutomaticWhiteBalanceMethod.cpp) |
| Auto Level Adjustment | Color Correction | ✅ | ✅ | ⭐⭐ | [Python](python/advanced/auto_level.py) / [C++](cpp/advanced/AutoLevelAndAutoContrast.cpp) |
| Illumination Correction | Image Correction | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/illumination_correction.py) / [C++](cpp/advanced/IlluminationCorrection.cpp) |
| Backlight Image Recovery | Image Recovery | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/backlight.py) / [C++](cpp/advanced/Inrbl.cpp) |
| 2D Gamma Correction | Image Correction | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/gamma_correction.py) / [C++](cpp/advanced/TwoDimensionalGamma.cpp) |

### 3. Image Dehazing Algorithms
| Algorithm | Category | Python | C++ | Difficulty | Code Reference |
|-----------|----------|---------|-----|------------|----------------|
| Dark Channel Prior | Image Dehazing | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/dark_channel.py) / [C++](cpp/advanced/DarkChannelPrior.cpp) |
| Guided Filter Dehazing | Image Dehazing | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/guided_filter.py) / [C++](cpp/advanced/GuidedFilterDehazing.cpp) |
| Median Filter Dehazing | Image Dehazing | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/median_filter.py) / [C++](cpp/advanced/MedianFilterFogRemoval.cpp) |
| Fast Single Image Dehazing | Image Dehazing | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/fast_defogging.py) / [C++](cpp/advanced/FastDefogging.cpp) |
| Real-time Video Dehazing | Image Dehazing | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/realtime_dehazing.py) / [C++](cpp/advanced/RealtimeDehazing.cpp) |

### 4. Advanced Filtering Algorithms
| Algorithm | Category | Python | C++ | Difficulty | Code Reference |
|-----------|----------|---------|-----|------------|----------------|
| Guided Filter | Image Filtering | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/guided_filter.py) / [C++](cpp/advanced/GuidedFilter.cpp) |
| Side Window Filter (Box) | Image Filtering | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/side_window.py) / [C++](cpp/advanced/BoxSideWindowFilter.cpp) |
| Side Window Filter (Median) | Image Filtering | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/side_window.py) / [C++](cpp/advanced/MedianSideWindowFilter.cpp) |
| Homomorphic Filter | Image Filtering | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/homomorphic.py) / [C++](cpp/advanced/HomomorphicFilter.cpp) |

### 5. Special Object Detection
| Algorithm | Category | Python | C++ | Difficulty | Code Reference |
|-----------|----------|---------|-----|------------|----------------|
| Rectangle Detection | Object Detection | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/rectangle_detection.py) / [C++](cpp/advanced/RectangleDetection.cpp) |
| License Plate Detection | Object Detection | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/license_plate.py) / [C++](cpp/advanced/LicensePlateDetection.cpp) |
| Color Cast Detection | Image Detection | ✅ | ✅ | ⭐⭐ | [Python](python/advanced/color_cast.py) / [C++](cpp/advanced/ColorCastDetection.cpp) |

### 6. Image Effect Algorithms
| Algorithm | Category | Python | C++ | Difficulty | Code Reference |
|-----------|----------|---------|-----|------------|----------------|
| Vintage Effect | Image Effect | ✅ | ✅ | ⭐⭐ | [Python](python/advanced/vintage_effect.py) / [C++](cpp/advanced/VintageEffect.cpp) |
| Motion Blur | Image Effect | ✅ | ✅ | ⭐⭐ | [Python](python/advanced/motion_blur.py) / [C++](cpp/advanced/MotionBlur.cpp) |
| Spherize Effect | Image Effect | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/spherize.py) / [C++](cpp/advanced/Spherize.cpp) |
| Skin Beauty | Image Effect | ✅ | ✅ | ⭐⭐ | [Python](python/advanced/skin_beauty.py) / [C++](cpp/advanced/SkinBeauty.cpp) |
| Unsharp Masking | Image Effect | ✅ | ✅ | ⭐⭐ | [Python](python/advanced/usm.py) / [C++](cpp/advanced/UnsharpMasking.cpp) |

## Usage

1. Clone the repository:
```bash
git clone https://github.com/GlimmerLab/IP101.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Choose your programming language:
   - Python users: Run Python files directly
   - C++ users: Configure OpenCV environment first

4. Run examples:
```bash
# Basic questions examples
python python/basic/color_operations.py 1  # Run question 1 in color operations
python python/basic/filtering.py 6         # Run question 6 in filtering

# Advanced algorithm examples
python python/advanced/retinex_msrcr.py
```

### C++ Project Building and Usage

1. Environment setup:
   - Install C++ compiler (such as GCC, Visual Studio, Clang)
   - Install CMake (version 3.10 or higher)
   - Install OpenCV library (version 4.0 or higher recommended)

2. Build the project:
```bash
# Create a build folder in the project root directory
mkdir build && cd build

# Configure with CMake (auto find OpenCV)
cmake ..

# Or manually specify OpenCV path
cmake -DOPENCV_DIR=/path/to/opencv/build ..

# Compile
cmake --build . --config Release
```

   **Tip**: You can also set the OpenCV path directly in the main CMakeLists.txt file:
   ```cmake
   # Open CMakeLists.txt, find the following section and uncomment, modify to your OpenCV path
   # set(OpenCV_DIR "D:/opencv/build")    # Windows example path
   # set(OpenCV_DIR "/usr/local/opencv4") # Linux example path
   ```

3. Run C++ examples:
```bash
# Run basic examples
./examples/basic_example

# Run specific algorithm tests
./examples/basic/color_operations_test
./examples/basic/filtering_test
```

4. Develop your own applications:
```cpp
// my_app.cpp
#include <opencv2/opencv.hpp>
#include <basic/color_operations.hpp>
#include <basic/filtering.hpp>

int main() {
    cv::Mat image = cv::imread("your_image.jpg");
    cv::Mat gray, filtered;

    // Use the grayscale conversion function from the library
    ip101::to_gray(image, gray);

    // Use Gaussian filter
    ip101::gaussian_filter(gray, filtered, 3, 1.0);

    cv::imshow("Filtered Image", filtered);
    cv::waitKey(0);

    return 0;
}
```

5. Compile custom applications:
```bash
g++ -std=c++17 my_app.cpp -o my_app -I/path/to/IP101/include `pkg-config --cflags --libs opencv4`
```

## Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Submit a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

IP101 references and draws inspiration from the following projects:

- [BBuf/Image-processing-algorithm](https://github.com/BBuf/Image-processing-algorithm)

- [gzr2017/ImageProcessing100Wen](https://github.com/gzr2017/ImageProcessing100Wen)

- [KuKuXia/Image_Processing_100_Questions](https://github.com/KuKuXia/Image_Processing_100_Questions)

- [ryoppippi/Gasyori100knock](https://github.com/ryoppippi/Gasyori100knock)

- [OpenCV](https://github.com/opencv/opencv)

- [scikit-image](https://github.com/scikit-image/scikit-image)

- [SimpleCV](https://github.com/sightmachine/SimpleCV)