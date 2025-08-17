# 图像矫正算法模块
set(CORRECTION_SOURCES
    automatic_white_balance.cpp
    auto_level.cpp
    illumination_correction.cpp
    backlight.cpp
    gamma_correction.cpp
)

# 创建静态库
add_library(ip101_correction STATIC ${CORRECTION_SOURCES})

# 链接库和包含目录
target_link_libraries(ip101_correction
    PUBLIC
    ${IP101_COMMON_LIBRARIES}
    ip101_basic
)

# 包含头文件目录
target_include_directories(ip101_correction
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/utils>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/advanced/correction>
    $<INSTALL_INTERFACE:include>
)