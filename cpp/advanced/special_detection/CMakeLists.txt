# 特殊目标检测算法模块
set(DETECTION_SOURCES
    rectangle_detection.cpp
    license_plate_detection.cpp
    color_cast_detection.cpp
)

# 创建静态库
add_library(ip101_detection STATIC ${DETECTION_SOURCES})

# 链接库和包含目录
target_link_libraries(ip101_detection
    PUBLIC
    ${IP101_COMMON_LIBRARIES}
    ip101_basic
)

# 包含头文件目录
target_include_directories(ip101_detection
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/utils>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/advanced/detection>
    $<INSTALL_INTERFACE:include>
)