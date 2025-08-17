# 图像去雾算法模块
set(DEFOGGING_SOURCES
    dark_channel.cpp
    guided_filter_dehazing.cpp
    median_filter.cpp
    fast_defogging.cpp
    realtime_dehazing.cpp
)

# 创建静态库
add_library(ip101_defogging STATIC ${DEFOGGING_SOURCES})

# 链接库和包含目录
target_link_libraries(ip101_defogging
    PUBLIC
    ${IP101_COMMON_LIBRARIES}
    ip101_basic
)

# 包含头文件目录
target_include_directories(ip101_defogging
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/utils>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/advanced/defogging>
    $<INSTALL_INTERFACE:include>
)