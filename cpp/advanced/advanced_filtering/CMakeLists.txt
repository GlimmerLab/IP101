# 高级滤波算法模块
set(FILTERING_SOURCES
    guided_filter.cpp
    side_window_filter.cpp
    homomorphic_filter.cpp
)

# 创建静态库
add_library(ip101_advanced_filtering STATIC ${FILTERING_SOURCES})

# 链接库和包含目录
target_link_libraries(ip101_advanced_filtering
    PUBLIC
    ${IP101_COMMON_LIBRARIES}
    ip101_basic
)

# 包含头文件目录
target_include_directories(ip101_advanced_filtering
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/utils>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/advanced/filtering>
    $<INSTALL_INTERFACE:include>
)