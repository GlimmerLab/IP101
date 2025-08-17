# 图像增强算法模块
set(ENHANCEMENT_SOURCES
    retinex_msrcr.cpp
    hdr.cpp
    adaptive_logarithmic_mapping.cpp
    multi_scale_detail_enhancement.cpp
    real_time_adaptive_contrast.cpp
    automatic_color_equalization.cpp
)

# 创建静态库
add_library(ip101_enhancement STATIC ${ENHANCEMENT_SOURCES})

# 链接库和包含目录
target_link_libraries(ip101_enhancement
    PUBLIC
    ${IP101_COMMON_LIBRARIES}
    ip101_basic
)

# 包含头文件目录
target_include_directories(ip101_enhancement
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/utils>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/advanced/enhancement>
    $<INSTALL_INTERFACE:include>
)