# 图像特效算法模块
set(EFFECTS_SOURCES
    vintage_effect.cpp
    motion_blur.cpp
    spherize.cpp
    skin_beauty.cpp
    unsharp_masking.cpp
    oil_painting_effect.cpp
    cartoon_effect.cpp
)

# 创建静态库
add_library(ip101_effects STATIC ${EFFECTS_SOURCES})

# 链接库和包含目录
target_link_libraries(ip101_effects
    PUBLIC
    ${IP101_COMMON_LIBRARIES}
    ip101_basic
)

# 包含头文件目录
target_include_directories(ip101_effects
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/utils>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/advanced/effects>
    $<INSTALL_INTERFACE:include>
)