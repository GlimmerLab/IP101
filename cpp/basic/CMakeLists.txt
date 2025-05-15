cmake_minimum_required(VERSION 3.10)
project(ip101_basic)

# 源文件列表（排除测试文件）
set(SOURCES
    color_operations.cpp
    edge_detection.cpp
    filtering.cpp
    feature_extraction.cpp
    frequency_domain.cpp
    image_compression.cpp
    image_enhancement.cpp
    image_inpainting.cpp
    image_matching.cpp
    image_features.cpp
    image_pyramid.cpp
    image_quality.cpp
    image_segmentation.cpp
    image_transform.cpp
    morphology.cpp
    object_detection.cpp
    super_resolution.cpp
    texture_analysis.cpp
    thinning.cpp
    connected_components.cpp
)

# 创建静态库
add_library(ip101_basic STATIC ${SOURCES})

# 链接库和包含目录（使用父级CMake定义的变量）
target_link_libraries(ip101_basic
    PUBLIC
    ${IP101_COMMON_LIBRARIES}
)

# 包含头文件目录
target_include_directories(ip101_basic
    PUBLIC
    ${IP101_COMMON_INCLUDES}
)

# 安装选项
install(TARGETS ip101_basic
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)