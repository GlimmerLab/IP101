# IP101 - 100 Questions in Image Processing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-blue.svg)](https://github.com/GlimmerLab/IP101)
[![Language](https://img.shields.io/badge/language-C%2B%2B%20%7C%20Python-orange.svg)](https://github.com/GlimmerLab/IP101)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green.svg)](https://opencv.org/)
[![CMake](https://img.shields.io/badge/CMake-3.10+-red.svg)](https://cmake.org/)

English | [简体中文](README.md)

IP101 is a comprehensive tutorial series focused on fundamental knowledge, operations, applications, and optimization in image processing. This series aims to help readers master core concepts and practical skills in image processing through 100 carefully designed questions.

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/GlimmerLab/IP101.git
cd IP101

# For Python users: Run directly
python python/basic/color_operations.py 1

# For C++ users: Build the project
mkdir build && cd build
cmake ..
cmake --build . --config Release
./examples/basic/color_operations_test
```

## 📋 Table of Contents
- [Project Structure](#project-structure)
- [Features](#features)
- [Basic Questions Categories](#basic-questions-categories)
- [Advanced Algorithm List](#advanced-algorithm-list)
- [Usage](#usage)
- [Contributing](#contributing)

## Project Structure

```
IP101/
├── include/            # Header files
│   ├── basic/         # Basic algorithm headers
│   └── advanced/      # Advanced algorithm headers
│       ├── correction/    # Image correction algorithms
│       ├── defogging/     # Image defogging algorithms
│       ├── detection/     # Special detection algorithms
│       ├── effects/       # Image effects algorithms
│       ├── enhancement/   # Image enhancement algorithms
│       └── filtering/     # Advanced filtering algorithms
├── cpp/                # C++ implementation
│   ├── basic/          # Basic questions code
│   └── advanced/       # Advanced algorithms code
│       ├── image_correction/    # Image correction
│       ├── image_defogging/     # Image defogging
│       ├── image_effects/       # Image effects
│       ├── image_enhancement/   # Image enhancement
│       ├── advanced_filtering/  # Advanced filtering
│       └── special_detection/   # Special detection
├── python/             # Python implementation
│   ├── basic/          # Basic questions code
│   ├── advanced/       # Advanced algorithms code
│   ├── image_processing/   # Image processing tools
│   └── tests/          # Test code
├── examples/           # Example code
│   ├── basic/          # Basic questions examples
│   └── advanced/       # Advanced algorithms examples
├── docs/               # Documentation
│   ├── algorithms/     # Algorithm documentation
│   ├── tutorials/      # Tutorials
│   └── optimization/   # Optimization techniques
├── gui/                # GUI interface
├── tests/              # Test code
├── utils/              # Utility functions
├── third_party/        # Third-party dependencies
│   ├── glfw/           # GLFW library
│   └── imgui/          # ImGui library
├── cmake/              # CMake configuration
└── assets/             # Asset files
```

## ✨ Project Highlights

### 🎯 Teaching-Oriented
- **Manual Implementation**: All algorithms are manually implemented without using OpenCV built-in functions to help understand algorithm principles
- **Progressive Learning**: Complete learning path from basics to advanced
- **Theory-Practice Integration**: Each algorithm comes with detailed mathematical principle explanations

### 🚀 Technical Advantages
- **High Performance**: C++ implementation supports SIMD optimization and multi-threading acceleration
- **Cross-Platform**: Supports Windows, Linux, and macOS
- **Easy Extension**: Modular design for easy addition of new algorithms

### 📚 Rich Content
- **100 Basic Questions**: Covers all areas of image processing
- **30+ Advanced Algorithms**: Includes latest research algorithms
- **Bilingual Support**: Chinese and English documentation and code comments

## Supported Languages

| Language | Status | Description |
|----------|---------|------------|
| Python | ✅ | Full support, includes solutions for all 100 questions |
| C++ | ✅ | Full support, includes solutions for all 100 questions |
| MATLAB | ❌ | Not supported yet |

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

## Advanced Algorithm List

### 1. Image Enhancement Algorithms
| Algorithm | Category | Python | C++ | Difficulty | Code Reference |
|-----------|----------|---------|-----|------------|----------------|
| Retinex MSRCR | Image Enhancement | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/retinex_msrcr.py) / [C++](cpp/advanced/image_enhancement/retinex_msrcr.cpp) |
| HDR | High Dynamic Range | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/hdr.py) / [C++](cpp/advanced/HDR.cpp) |
| Adaptive Logarithmic Mapping | High Dynamic Range | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/adaptive_logarithmic_mapping.py) / [C++](cpp/advanced/image_enhancement/adaptive_logarithmic_mapping.cpp) |
| Multi-scale Detail Enhancement | Image Enhancement | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/multi_scale_detail_enhancement.py) / [C++](cpp/advanced/image_enhancement/multi_scale_detail_enhancement.cpp) |
| Real-time Adaptive Contrast | Image Enhancement | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/real_time_adaptive_contrast.py) / [C++](cpp/advanced/image_enhancement/real_time_adaptive_contrast.cpp) |
| Automatic Color Equalization (ACE) | Color Enhancement | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/automatic_color_equalization.py) / [C++](cpp/advanced/image_enhancement/automatic_color_equalization.cpp) |

### 2. Image Correction Algorithms
| Algorithm | Category | Python | C++ | Difficulty | Code Reference |
|-----------|----------|---------|-----|------------|----------------|
| Auto White Balance | Color Correction | ✅ | ✅ | ⭐⭐ | [Python](python/advanced/automatic_white_balance.py) / [C++](cpp/advanced/image_correction/automatic_white_balance.cpp) |
| Auto Level Adjustment | Color Correction | ✅ | ✅ | ⭐⭐ | [Python](python/advanced/auto_level_adjustment.py) / [C++](cpp/advanced/image_correction/auto_level.cpp) |
| Illumination Correction | Image Correction | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/illumination_correction.py) / [C++](cpp/advanced/image_correction/illumination_correction.cpp) |
| Backlight Image Recovery | Image Recovery | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/backlight_correction.py) / [C++](cpp/advanced/image_correction/backlight.cpp) |
| 2D Gamma Correction | Image Correction | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/gamma_correction.py) / [C++](cpp/advanced/image_correction/gamma_correction.cpp) |

### 3. Image Dehazing Algorithms
| Algorithm | Category | Python | C++ | Difficulty | Code Reference |
|-----------|----------|---------|-----|------------|----------------|
| Dark Channel Prior | Image Dehazing | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/dark_channel_dehazing.py) / [C++](cpp/advanced/image_defogging/dark_channel.cpp) |
| Guided Filter Dehazing | Image Dehazing | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/guided_filter.py) / [C++](cpp/advanced/image_defogging/guided_filter.cpp) |
| Median Filter Dehazing | Image Dehazing | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/median_filter_defogging.py) / [C++](cpp/advanced/image_defogging/median_filter.cpp) |
| Fast Single Image Dehazing | Image Dehazing | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/fast_defogging.py) / [C++](cpp/advanced/image_defogging/fast_defogging.cpp) |
| Real-time Video Dehazing | Image Dehazing | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/realtime_dehazing.py) / [C++](cpp/advanced/image_defogging/realtime_dehazing.cpp) |

### 4. Advanced Filtering Algorithms
| Algorithm | Category | Python | C++ | Difficulty | Code Reference |
|-----------|----------|---------|-----|------------|----------------|
| Guided Filter | Image Filtering | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/guided_filter.py) / [C++](cpp/advanced/advanced_filtering/guided_filter.cpp) |
| Side Window Filter (Box) | Image Filtering | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/side_window_filter.py) / [C++](cpp/advanced/advanced_filtering/side_window_filter.cpp) |
| Side Window Filter (Median) | Image Filtering | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/side_window_filter.py) / [C++](cpp/advanced/advanced_filtering/side_window_filter.cpp) |
| Homomorphic Filter | Image Filtering | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/homomorphic_filter.py) / [C++](cpp/advanced/advanced_filtering/homomorphic_filter.cpp) |

### 5. Special Object Detection
| Algorithm | Category | Python | C++ | Difficulty | Code Reference |
|-----------|----------|---------|-----|------------|----------------|
| Rectangle Detection | Object Detection | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/rectangle_detection.py) / [C++](cpp/advanced/special_detection/rectangle_detection.cpp) |
| License Plate Detection | Object Detection | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/license_plate_detection.py) / [C++](cpp/advanced/special_detection/license_plate_detection.cpp) |
| Color Cast Detection | Image Detection | ✅ | ✅ | ⭐⭐ | [Python](python/advanced/color_cast_detection.py) / [C++](cpp/advanced/special_detection/color_cast_detection.cpp) |

### 6. Image Effect Algorithms
| Algorithm | Category | Python | C++ | Difficulty | Code Reference |
|-----------|----------|---------|-----|------------|----------------|
| Vintage Effect | Image Effect | ✅ | ✅ | ⭐⭐ | [Python](python/advanced/vintage_effect.py) / [C++](cpp/advanced/image_effects/vintage_effect.cpp) |
| Motion Blur | Image Effect | ✅ | ✅ | ⭐⭐ | [Python](python/advanced/motion_blur_effect.py) / [C++](cpp/advanced/image_effects/motion_blur.cpp) |
| Spherize Effect | Image Effect | ✅ | ✅ | ⭐⭐⭐ | [Python](python/advanced/spherize_effect.py) / [C++](cpp/advanced/image_effects/spherize.cpp) |
| Skin Beauty | Image Effect | ✅ | ✅ | ⭐⭐ | [Python](python/advanced/skin_beauty.py) / [C++](cpp/advanced/image_effects/skin_beauty.cpp) |
| Unsharp Masking | Image Effect | ✅ | ✅ | ⭐⭐ | [Python](python/advanced/unsharp_masking.py) / [C++](cpp/advanced/image_effects/unsharp_masking.cpp) |
| Oil Painting Effect | Image Effect | ✅ | ✅ | ⭐⭐ | [Python](python/advanced/oil_painting_effect.py) / [C++](cpp/advanced/image_effects/oil_painting_effect.cpp) |
| Cartoon Effect | Image Effect | ✅ | ✅ | ⭐⭐ | [Python](python/advanced/cartoon_effect.py) / [C++](cpp/advanced/image_effects/cartoon_effect.cpp) |

## 📚 Documentation & Tutorials

### 📖 Algorithm Documentation (`docs/algorithms/`)
Comprehensive technical documentation including:
- **Algorithm Principles**: Mathematical derivation and core concepts
- **Complexity Analysis**: Time and space complexity evaluation
- **Application Scenarios**: Real-world use cases and best practices
- **Parameter Tuning**: Algorithm parameter selection and optimization tips

### 🎓 Learning Tutorials (`docs/tutorials/`)
Complete learning path from beginner to expert:
- **Environment Setup**: Development environment configuration guide
- **Basic Introduction**: Image processing fundamental concepts
- **Practical Cases**: Step-by-step algorithm implementation tutorials
- **Performance Analysis**: Code performance evaluation and optimization

### ⚡ Performance Optimization (`docs/optimization/`)
Advanced optimization techniques guide:
- **Multi-threading**: OpenMP and thread pool optimization
- **SIMD Vectorization**: CPU instruction set optimization
- **Memory Management**: Cache-friendly data structures
- **Algorithm Improvements**: Mathematical optimization and approximation algorithms

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

**Important Notes**:
- Make sure to run executables from the correct build directory (e.g., `build/Release/` or `build/Debug/`)
- **Windows users pay special attention**: You need to copy OpenCV DLL files (such as `opencv_world4xx.dll`) to the executable directory, or add them to the system PATH environment variable
- If you encounter "opencv_world4xx.dll not found" error, check if OpenCV's bin directory is in PATH, or manually copy DLL files to the program directory

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

## ❓ FAQ

### Q: Why manual implementation instead of using OpenCV built-in functions?
A: Manual implementation helps deeply understand algorithm principles, which is the teaching feature of this project. In actual projects, you can choose to use OpenCV built-in functions as needed.

### Q: How to choose the right programming language?
A:
- **Python**: Suitable for rapid prototyping and algorithm verification
- **C++**: Suitable for production environments with high performance requirements

### Q: What to do if encountering DLL errors on Windows?
A: Please refer to the Windows-specific tips in the [Usage](#usage) section to ensure OpenCV DLL files are in the correct location.

### Q: How to contribute code?
A: Welcome to submit Issues and Pull Requests! Please refer to the contribution guidelines below.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Submit a Pull Request

### Contribution Types
- 🐛 Bug fixes
- ✨ New features
- 📚 Documentation improvements
- 🎨 Code optimization
- 🧪 Test cases

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

IP101 references and draws inspiration from the following projects:

### 🖼️ Image Processing Algorithm References
- [BBuf/Image-processing-algorithm](https://github.com/BBuf/Image-processing-algorithm)
- [gzr2017/ImageProcessing100Wen](https://github.com/gzr2017/ImageProcessing100Wen)
- [KuKuXia/Image_Processing_100_Questions](https://github.com/KuKuXia/Image_Processing_100_Questions)
- [ryoppippi/Gasyori100knock](https://github.com/ryoppippi/Gasyori100knock)

### 🔧 Core Dependencies
- [OpenCV](https://github.com/opencv/opencv) - Computer Vision Library
- [scikit-image](https://github.com/scikit-image/scikit-image) - Python Image Processing Library
- [SimpleCV](https://github.com/sightmachine/SimpleCV) - Computer Vision Framework

### 🖥️ GUI Interface Dependencies
- [GLFW](https://github.com/glfw/glfw) - Cross-platform OpenGL context and window management library
- [Dear ImGui](https://github.com/ocornut/imgui) - Lightweight immediate mode graphical user interface library

### 📚 Learning Resources
- [imageshop](https://www.cnblogs.com/imageshop) - Image Processing Technology Blog